"""Fine-tune DeBERTa-v3-small for ordinal automated essay scoring.

The command supports a short, resource-measured smoke test and full fold runs.
It uses the immutable project folds and writes validation/test predictions under
``artifacts/deberta``.  No hidden-test labels or external essay labels are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import transformers
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split

try:
    from src.config import (
        ARTIFACTS_DIR,
        ID_COLUMN,
        N_SPLITS,
        RANDOM_SEED,
        SCORE_MAX,
        SCORE_MIN,
        TARGET_COLUMN,
        TEXT_COLUMN,
    )
    from src.metrics import apply_ordered_thresholds, quadratic_weighted_kappa
    from src.train_tfidf import file_sha256, validate_and_load_data
except ModuleNotFoundError:  # Support ``python src/train_deberta_ordinal.py``.
    from config import (  # type: ignore[no-redef]
        ARTIFACTS_DIR,
        ID_COLUMN,
        N_SPLITS,
        RANDOM_SEED,
        SCORE_MAX,
        SCORE_MIN,
        TARGET_COLUMN,
        TEXT_COLUMN,
    )
    from metrics import (  # type: ignore[no-redef]
        apply_ordered_thresholds,
        quadratic_weighted_kappa,
    )
    from train_tfidf import file_sha256, validate_and_load_data  # type: ignore[no-redef]


DEFAULT_MODEL = "microsoft/deberta-v3-small"
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "deberta"
ORDINAL_BOUNDARIES = SCORE_MAX - SCORE_MIN


@dataclass(frozen=True)
class TrainingConfig:
    """Serializable settings for one fold training run."""

    model_name: str
    fold: int
    max_length: int
    tail_fraction: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    max_steps: int
    encoder_learning_rate: float
    head_learning_rate: float
    weight_decay: float
    warmup_ratio: float
    regression_loss_weight: float
    dropout: float
    max_grad_norm: float
    initial_loss_scale: float
    mixed_precision: bool
    gradient_checkpointing: bool
    seed: int
    validation_limit: int
    predict_test: bool
    save_model: bool


class EncodedEssayDataset(Dataset[dict[str, Any]]):
    """Store variable-length token IDs and optional integer essay scores."""

    def __init__(
        self,
        encodings: dict[str, list[list[int]]],
        labels: np.ndarray | None = None,
    ) -> None:
        self.encodings = encodings
        self.labels = labels
        lengths = {len(values) for values in encodings.values()}
        if len(lengths) != 1:
            raise ValueError("tokenizer outputs have inconsistent row counts")
        self.length = lengths.pop()
        if labels is not None and len(labels) != self.length:
            raise ValueError("labels and tokenized essays have different lengths")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        row: dict[str, Any] = {
            key: values[index] for key, values in self.encodings.items()
        }
        if self.labels is not None:
            row["labels"] = int(self.labels[index])
        return row


class DynamicPaddingCollator:
    """Pad each batch only to its longest sequence, rounded for GPU kernels."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = None
        if "labels" in rows[0]:
            labels = torch.tensor(
                [row.pop("labels") for row in rows],
                dtype=torch.long,
            )
        batch = self.tokenizer.pad(
            rows,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        if labels is not None:
            batch["labels"] = labels
        return batch


def inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    """Return numerically stable parameters whose softplus equals ``values``."""

    return values + torch.log(-torch.expm1(-values))


def empirical_cutpoints(labels: np.ndarray) -> np.ndarray:
    """Initialize ordered logits to the training fold's class distribution."""

    probabilities = np.array(
        [(labels > boundary).mean() for boundary in range(SCORE_MIN, SCORE_MAX)],
        dtype=np.float64,
    )
    probabilities = np.clip(probabilities, 1e-4, 1.0 - 1e-4)
    cutpoints = -np.log(probabilities / (1.0 - probabilities))
    if not np.all(np.diff(cutpoints) > 0):
        raise ValueError("empirical ordinal cutpoints are not strictly ordered")
    return cutpoints.astype(np.float32)


class OrdinalDeberta(nn.Module):
    """DeBERTa encoder with a monotonic cumulative-link ordinal head."""

    def __init__(
        self,
        model_name: str,
        initial_cutpoints: np.ndarray,
        dropout: float,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True,
        )
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.encoder.config, "use_cache"):
            self.encoder.config.use_cache = False
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.latent_score = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.latent_score.weight)
        nn.init.zeros_(self.latent_score.bias)

        cuts = torch.as_tensor(initial_cutpoints, dtype=torch.float32)
        if cuts.shape != (ORDINAL_BOUNDARIES,):
            raise ValueError("initial_cutpoints has the wrong shape")
        self.cut_base = nn.Parameter(cuts[:1].clone())
        self.cut_delta_raw = nn.Parameter(inverse_softplus(torch.diff(cuts)))

    def ordered_cutpoints(self) -> torch.Tensor:
        """Construct five strictly increasing learned decision boundaries."""

        positive_deltas = functional.softplus(self.cut_delta_raw) + 1e-4
        return torch.cat(
            [self.cut_base, self.cut_base + torch.cumsum(positive_deltas, dim=0)]
        )

    def forward(self, **inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cumulative logits and a continuous expected score in 1..6."""

        attention_mask = inputs["attention_mask"]
        encoded = self.encoder(**inputs).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        latent = self.latent_score(self.dropout(pooled)).squeeze(-1)
        logits = latent.unsqueeze(1) - self.ordered_cutpoints().unsqueeze(0)
        expected_score = SCORE_MIN + torch.sigmoid(logits).sum(dim=1)
        return logits, expected_score


def parse_args() -> argparse.Namespace:
    """Parse one smoke-test or full-fold training configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--fold", type=int, default=0, choices=range(N_SPLITS))
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.25,
        help=(
            "For overlength essays, reserve this fraction of the token budget "
            "for the conclusion and retain the rest from the beginning; use 0 "
            "for ordinary right truncation."
        ),
    )
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optimizer-step cap; 0 runs all requested epochs.",
    )
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--regression-loss-weight", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--initial-loss-scale", type=float, default=32_768.0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=0,
        help="Stratified validation-row cap for smoke tests; 0 evaluates the full fold.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable float16 autocast and gradient scaling.",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without promising strict MPS determinism."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(requested: str) -> torch.device:
    """Select MPS when available and fail clearly when it was explicitly required."""

    if requested == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "mps":
        raise RuntimeError("MPS was requested but is unavailable in this process")
    return torch.device("cpu")


def tokenize_texts(
    tokenizer: Any,
    texts: pd.Series,
    max_length: int,
    tail_fraction: float,
    split_name: str,
) -> tuple[dict[str, list[list[int]]], dict[str, float | int]]:
    """Tokenize with dynamic padding and optional beginning/conclusion retention."""

    raw_encoded = tokenizer(
        texts.astype(str).tolist(),
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    single_special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    single_budget = max_length - single_special_tokens
    if single_budget < 2:
        raise ValueError("max_length leaves no room for essay tokens")

    include_token_types = "token_type_ids" in tokenizer.model_input_names
    encodings: dict[str, list[list[int]]] = {
        "input_ids": [],
        "attention_mask": [],
    }
    if include_token_types:
        encodings["token_type_ids"] = []
    original_lengths: list[int] = []
    truncated_rows = 0
    for token_ids in raw_encoded["input_ids"]:
        original_lengths.append(len(token_ids))
        if len(token_ids) <= single_budget:
            first_tokens = token_ids
            second_tokens = None
        elif tail_fraction > 0.0:
            truncated_rows += 1
            tail_tokens = max(1, int(single_budget * tail_fraction))
            head_tokens = single_budget - tail_tokens
            first_tokens = token_ids[:head_tokens] + token_ids[-tail_tokens:]
            second_tokens = None
        else:
            truncated_rows += 1
            first_tokens = token_ids[:single_budget]
            second_tokens = None

        input_ids = tokenizer.build_inputs_with_special_tokens(
            first_tokens,
            second_tokens,
        )
        if len(input_ids) > max_length:
            raise RuntimeError("token construction exceeded max_length")
        encodings["input_ids"].append(input_ids)
        encodings["attention_mask"].append([1] * len(input_ids))
        if include_token_types:
            encodings["token_type_ids"].append(
                tokenizer.create_token_type_ids_from_sequences(
                    first_tokens,
                    second_tokens,
                )
            )

    stats: dict[str, float | int] = {
        "rows": len(original_lengths),
        "truncated_rows": truncated_rows,
        "truncated_percent": 100.0 * truncated_rows / max(1, len(original_lengths)),
        "original_token_median": float(np.median(original_lengths)),
        "original_token_p95": float(np.quantile(original_lengths, 0.95)),
        "original_token_max": int(max(original_lengths, default=0)),
    }
    print(
        f"{split_name} tokens: median={stats['original_token_median']:.0f}, "
        f"p95={stats['original_token_p95']:.0f}, "
        f"truncated={truncated_rows:,}/{len(original_lengths):,} "
        f"({stats['truncated_percent']:.1f}%)",
        flush=True,
    )
    return encodings, stats


def move_inputs(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """Move model inputs and optional labels to the selected device."""

    labels = batch.pop("labels", None)
    inputs = {key: value.to(device) for key, value in batch.items()}
    return inputs, labels.to(device) if labels is not None else None


def ordinal_targets(labels: torch.Tensor) -> torch.Tensor:
    """Encode scores 1..6 as five cumulative binary targets."""

    boundaries = torch.arange(
        SCORE_MIN,
        SCORE_MAX,
        device=labels.device,
        dtype=labels.dtype,
    )
    return (labels.unsqueeze(1) > boundaries.unsqueeze(0)).to(torch.float32)


def predict(
    model: OrdinalDeberta,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    mixed_precision: bool,
) -> np.ndarray:
    """Return continuous expected-score predictions for one data loader."""

    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            inputs, _ = move_inputs(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision,
            ):
                _, expected_score = model(**inputs)
            predictions.append(expected_score.float().cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def mps_memory_megabytes(device: torch.device) -> float | None:
    """Return current MPS allocation when supported by this PyTorch build."""

    if device.type != "mps" or not hasattr(torch.mps, "current_allocated_memory"):
        return None
    return float(torch.mps.current_allocated_memory() / (1024**2))


def validate_prediction_vector(
    name: str,
    predictions: np.ndarray,
    expected_rows: int,
) -> None:
    """Fail before persistence when a prediction vector is malformed."""

    if predictions.shape != (expected_rows,):
        raise RuntimeError(
            f"{name} predictions have shape {predictions.shape}; "
            f"expected ({expected_rows},)"
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{name} predictions contain NaN or Inf")
    tolerance = 1e-5
    if (
        float(predictions.min()) < SCORE_MIN - tolerance
        or float(predictions.max()) > SCORE_MAX + tolerance
    ):
        raise RuntimeError(f"{name} raw predictions fall outside [{SCORE_MIN}, {SCORE_MAX}]")


def evaluate_validation(
    model: OrdinalDeberta,
    loader: DataLoader[dict[str, torch.Tensor]],
    labels: np.ndarray,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[np.ndarray, float, float]:
    """Predict one validation split and calculate fixed-threshold diagnostics."""

    predictions = predict(model, loader, device, mixed_precision)
    validate_prediction_vector("validation", predictions, len(labels))
    fixed_predictions = apply_ordered_thresholds(predictions)
    qwk = quadratic_weighted_kappa(labels, fixed_predictions)
    rmse = float(np.sqrt(np.mean(np.square(predictions - labels))))
    return predictions, qwk, rmse


def make_run_name(config: TrainingConfig) -> tuple[str, str]:
    """Build a readable, collision-resistant directory name from all settings."""

    serialized = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    run_kind = f"s{config.max_steps}" if config.max_steps else f"e{config.epochs}"
    tail_percent = int(round(100 * config.tail_fraction))
    run_name = (
        f"fold{config.fold}_{run_kind}_len{config.max_length}_ht{tail_percent}_"
        f"b{config.train_batch_size}a{config.gradient_accumulation_steps}_"
        f"seed{config.seed}_{config_hash}"
    )
    return run_name, config_hash


def main() -> None:
    """Train one immutable fold and persist validation diagnostics."""

    args = parse_args()
    if args.max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if args.validation_limit < 0:
        raise ValueError("validation_limit must be non-negative")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.log_every < 1:
        raise ValueError("log_every must be positive")
    if args.max_length < 32:
        raise ValueError("max_length is unexpectedly small")
    if args.initial_loss_scale <= 0:
        raise ValueError("initial_loss_scale must be positive")
    if not 0.0 <= args.tail_fraction < 0.5:
        raise ValueError("tail_fraction must be in [0, 0.5)")

    config = TrainingConfig(
        model_name=args.model_name,
        fold=args.fold,
        max_length=args.max_length,
        tail_fraction=args.tail_fraction,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        max_steps=args.max_steps,
        encoder_learning_rate=args.encoder_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        regression_loss_weight=args.regression_loss_weight,
        dropout=args.dropout,
        max_grad_norm=args.max_grad_norm,
        initial_loss_scale=args.initial_loss_scale,
        mixed_precision=not args.no_mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        validation_limit=args.validation_limit,
        predict_test=args.predict_test,
        save_model=args.save_model,
    )
    run_name, config_hash = make_run_name(config)
    output_dir = args.output_dir / run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite an existing DeBERTa run: {output_dir}"
        )
    set_seed(config.seed + config.fold)
    torch.set_float32_matmul_precision("high")
    device = select_device(args.device)
    mixed_precision = config.mixed_precision and device.type == "mps"
    if device.type == "cpu" and config.mixed_precision:
        print("Mixed precision disabled because the selected device is CPU.", flush=True)

    started = time.perf_counter()
    train, test, sample_submission = validate_and_load_data()
    training = train["fold"].to_numpy(dtype=int) != config.fold
    validation = ~training
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int64)
    if not training.any() or not validation.any():
        raise RuntimeError("selected fold produced an empty train or validation split")
    evaluation = validation.copy()
    validation_indices = np.flatnonzero(validation)
    if 0 < args.validation_limit < len(validation_indices):
        _, selected_indices = train_test_split(
            validation_indices,
            test_size=args.validation_limit,
            random_state=config.seed + config.fold,
            stratify=y[validation_indices],
        )
        evaluation[:] = False
        evaluation[selected_indices] = True

    print(
        f"Loading tokenizer {config.model_name!r}; device={device}, "
        f"train={training.sum():,}, validation={evaluation.sum():,}"
        f"/{validation.sum():,}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    train_encodings, train_token_stats = tokenize_texts(
        tokenizer,
        train.loc[training, TEXT_COLUMN],
        config.max_length,
        config.tail_fraction,
        "train",
    )
    validation_encodings, validation_token_stats = tokenize_texts(
        tokenizer,
        train.loc[evaluation, TEXT_COLUMN],
        config.max_length,
        config.tail_fraction,
        "validation",
    )
    collator = DynamicPaddingCollator(tokenizer)
    generator = torch.Generator().manual_seed(config.seed + config.fold)
    train_loader = DataLoader(
        EncodedEssayDataset(train_encodings, y[training]),
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        EncodedEssayDataset(validation_encodings, y[evaluation]),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    print("Loading base encoder weights...", flush=True)
    model = OrdinalDeberta(
        model_name=config.model_name,
        initial_cutpoints=empirical_cutpoints(y[training]),
        dropout=config.dropout,
        gradient_checkpointing=config.gradient_checkpointing,
    ).to(device)
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = [
        *model.latent_score.parameters(),
        model.cut_base,
        model.cut_delta_raw,
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": config.encoder_learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": head_parameters,
                "lr": config.head_learning_rate,
                "weight_decay": 0.0,
            },
        ]
    )
    steps_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    planned_steps = steps_per_epoch * config.epochs
    total_steps = min(planned_steps, config.max_steps) if config.max_steps else planned_steps
    if total_steps < 1:
        raise ValueError("training configuration produces zero optimizer steps")
    warmup_steps = int(round(config.warmup_ratio * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        init_scale=config.initial_loss_scale,
        enabled=mixed_precision,
    )

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    attempted_optimizer_steps = 0
    skipped_optimizer_steps = 0
    micro_step = 0
    loss_sum = 0.0
    examples_seen = 0
    training_started = time.perf_counter()
    training_compute_seconds = 0.0
    stop_training = False
    epoch_metrics: list[dict[str, float | int]] = []
    best_epoch = 0
    best_selection_key = (-math.inf, -math.inf)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_predictions: np.ndarray | None = None
    last_completed_epoch = 0
    for epoch in range(config.epochs):
        epoch_training_started = time.perf_counter()
        model.train()
        for batch_index, batch in enumerate(train_loader):
            inputs, labels = move_inputs(batch, device)
            if labels is None:
                raise RuntimeError("training batch unexpectedly has no labels")
            group_start = (
                batch_index // config.gradient_accumulation_steps
            ) * config.gradient_accumulation_steps
            accumulation_group_size = min(
                config.gradient_accumulation_steps,
                len(train_loader) - group_start,
            )
            group_start_row = group_start * config.train_batch_size
            accumulation_group_examples = min(
                accumulation_group_size * config.train_batch_size,
                len(train_loader.dataset) - group_start_row,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision,
            ):
                logits, expected_score = model(**inputs)
                cumulative_targets = ordinal_targets(labels)
                ordinal_loss = functional.binary_cross_entropy_with_logits(
                    logits,
                    cumulative_targets,
                )
                regression_loss = functional.smooth_l1_loss(
                    expected_score.float(),
                    labels.float(),
                )
                loss = ordinal_loss + config.regression_loss_weight * regression_loss
                scaled_loss = (
                    loss * int(labels.shape[0]) / accumulation_group_examples
                )
            scaler.scale(scaled_loss).backward()
            micro_step += 1
            examples_seen += int(labels.shape[0])
            loss_sum += float(loss.detach().cpu())

            end_of_epoch = batch_index + 1 == len(train_loader)
            accumulation_complete = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
            )
            if accumulation_complete or end_of_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer.zero_grad(set_to_none=True)
                attempted_optimizer_steps += 1
                step_succeeded = (not mixed_precision) or scale_after >= scale_before
                if step_succeeded:
                    scheduler.step()
                    optimizer_step += 1
                else:
                    skipped_optimizer_steps += 1
                    print(
                        f"Skipped optimizer update after non-finite gradients; "
                        f"loss scale {scale_before:g} -> {scale_after:g}",
                        flush=True,
                    )
                if step_succeeded and (
                    optimizer_step % args.log_every == 0 or optimizer_step == 1
                ):
                    elapsed = training_compute_seconds + (
                        time.perf_counter() - epoch_training_started
                    )
                    memory = mps_memory_megabytes(device)
                    memory_text = f", MPS={memory:.0f} MiB" if memory is not None else ""
                    print(
                        f"step {optimizer_step}/{total_steps}, "
                        f"loss={loss_sum / micro_step:.5f}, "
                        f"examples/s={examples_seen / elapsed:.2f}{memory_text}",
                        flush=True,
                    )
                if attempted_optimizer_steps >= total_steps:
                    stop_training = True
                    break
        training_compute_seconds += time.perf_counter() - epoch_training_started
        last_completed_epoch = epoch + 1
        if not config.max_steps:
            print(f"Evaluating validation fold after epoch {epoch + 1}...", flush=True)
            epoch_predictions, epoch_qwk, epoch_rmse = evaluate_validation(
                model,
                validation_loader,
                y[evaluation],
                device,
                mixed_precision,
            )
            epoch_metrics.append(
                {
                    "epoch": epoch + 1,
                    "optimizer_steps": optimizer_step,
                    "validation_fixed_qwk": epoch_qwk,
                    "validation_rmse": epoch_rmse,
                }
            )
            print(
                f"epoch {epoch + 1}: validation QWK={epoch_qwk:.5f}, "
                f"RMSE={epoch_rmse:.5f}",
                flush=True,
            )
            selection_key = (epoch_qwk, -epoch_rmse)
            if selection_key > best_selection_key:
                best_selection_key = selection_key
                best_epoch = epoch + 1
                best_validation_predictions = epoch_predictions
                if config.epochs > 1:
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }
        if stop_training:
            break
    training_wall_seconds = time.perf_counter() - training_started
    training_seconds = training_compute_seconds

    if attempted_optimizer_steps != total_steps:
        raise RuntimeError(
            f"completed {attempted_optimizer_steps} optimizer attempts, expected "
            f"{total_steps}; successful={optimizer_step}, "
            f"skipped={skipped_optimizer_steps}"
        )
    if config.max_steps:
        print("Evaluating validation fold...", flush=True)
        validation_predictions, validation_qwk, validation_rmse = evaluate_validation(
            model,
            validation_loader,
            y[evaluation],
            device,
            mixed_precision,
        )
        best_epoch = last_completed_epoch
        epoch_metrics.append(
            {
                "epoch": last_completed_epoch,
                "optimizer_steps": optimizer_step,
                "validation_fixed_qwk": validation_qwk,
                "validation_rmse": validation_rmse,
            }
        )
    else:
        if best_validation_predictions is None or best_epoch < 1:
            raise RuntimeError("full training completed without validation metrics")
        if best_epoch != last_completed_epoch:
            if best_state is None:
                raise RuntimeError("best checkpoint was not retained")
            model.load_state_dict(best_state)
            validation_predictions, validation_qwk, validation_rmse = (
                evaluate_validation(
                    model,
                    validation_loader,
                    y[evaluation],
                    device,
                    mixed_precision,
                )
            )
        else:
            validation_predictions = best_validation_predictions
            validation_qwk = float(best_selection_key[0])
            validation_rmse = float(-best_selection_key[1])

    fixed_predictions = apply_ordered_thresholds(validation_predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_ids = train.loc[evaluation, ID_COLUMN].to_numpy()
    if len(np.unique(validation_ids)) != len(validation_ids):
        raise RuntimeError("validation IDs are not unique")
    validation_frame = train.loc[
        evaluation,
        [ID_COLUMN, TARGET_COLUMN, "fold", "group_id"],
    ].copy()
    if not np.array_equal(validation_frame[ID_COLUMN].to_numpy(), validation_ids):
        raise RuntimeError("validation ID order changed before persistence")
    validation_frame["pred_raw"] = validation_predictions
    validation_frame["pred_fixed"] = fixed_predictions.astype(int)
    validation_path = output_dir / "validation_predictions.csv"
    validation_frame.to_csv(validation_path, index=False, float_format="%.10f")

    test_path = None
    if args.predict_test:
        test_encodings, test_token_stats = tokenize_texts(
            tokenizer,
            test[TEXT_COLUMN],
            config.max_length,
            config.tail_fraction,
            "test",
        )
        test_loader = DataLoader(
            EncodedEssayDataset(test_encodings),
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        test_predictions = predict(model, test_loader, device, mixed_precision)
        validate_prediction_vector("test", test_predictions, len(test))
        if test[ID_COLUMN].duplicated().any():
            raise RuntimeError("test IDs are not unique")
        if not test[ID_COLUMN].equals(sample_submission[ID_COLUMN]):
            raise RuntimeError("test IDs no longer match sample-submission order")
        test_frame = test[[ID_COLUMN]].copy()
        test_frame["pred_raw"] = test_predictions
        test_path = output_dir / "test_predictions.csv"
        test_frame.to_csv(test_path, index=False, float_format="%.10f")

    model_path = None
    if args.save_model:
        model_path = output_dir / "model_state.pt"
        model.to("cpu")
        torch.save(model.state_dict(), model_path)

    total_seconds = time.perf_counter() - started
    summary = {
        "run_name": run_name,
        "config_hash": config_hash,
        "config": asdict(config),
        "device": str(device),
        "mixed_precision_active": mixed_precision,
        "tokenization": {
            "train": train_token_stats,
            "validation": validation_token_stats,
            "test": test_token_stats if args.predict_test else None,
        },
        "model_revision": getattr(model.encoder.config, "_commit_hash", None),
        "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get(
            "_commit_hash"
        ),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "train_rows": int(training.sum()),
        "validation_fold_rows": int(validation.sum()),
        "validation_rows": int(evaluation.sum()),
        "optimizer_steps": optimizer_step,
        "attempted_optimizer_steps": attempted_optimizer_steps,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "micro_steps": micro_step,
        "examples_seen": examples_seen,
        "training_seconds": training_seconds,
        "training_wall_seconds": training_wall_seconds,
        "seconds_per_optimizer_step": training_seconds / optimizer_step,
        "examples_per_second": examples_seen / training_seconds,
        "validation_fixed_qwk": validation_qwk,
        "validation_rmse": validation_rmse,
        "validation_prediction_min": float(validation_predictions.min()),
        "validation_prediction_max": float(validation_predictions.max()),
        "best_epoch": best_epoch,
        "epoch_metrics": epoch_metrics,
        "learned_cutpoints": model.ordered_cutpoints().detach().cpu().tolist(),
        "total_runtime_seconds": total_seconds,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "artifacts": {
            "validation_predictions": str(validation_path),
            "validation_predictions_sha256": file_sha256(validation_path),
            "test_predictions": str(test_path) if test_path else None,
            "test_predictions_sha256": file_sha256(test_path) if test_path else None,
            "model_state": str(model_path) if model_path else None,
            "model_state_sha256": file_sha256(model_path) if model_path else None,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "DeBERTa ordinal fold complete\n"
        f"  Validation fixed QWK: {validation_qwk:.5f}\n"
        f"  Validation RMSE:      {validation_rmse:.5f}\n"
        f"  Step time:            {summary['seconds_per_optimizer_step']:.3f}s\n"
        f"  Training runtime:     {training_seconds / 60:.1f} min\n"
        f"  Summary:              {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
