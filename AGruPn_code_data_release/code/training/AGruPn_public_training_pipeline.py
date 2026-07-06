"""
Public training protocol for AGruPn.

This file documents the reproducible training workflow and the public data
interfaces used in the revised manuscript. The complete proprietary model
blocks, including the unpublished AGruPn core architecture and some symbolic
physics modules reserved for follow-up research, are intentionally omitted from
this public release.

The purpose of this script is to make the experimental protocol transparent:
data loading, sequence construction, split policy, metric computation, and the
places where the released notebooks implement partial model variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RELEASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = RELEASE_DIR / "data" / "prediction_results"


@dataclass(frozen=True)
class TrainingConfig:
    """Main settings reported in the manuscript and used by public scripts."""

    context_length: int = 9
    forecast_horizon: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 64
    max_epochs: int = 200
    learning_rate: float = 1e-3
    random_seed: int = 42
    quantiles: tuple[float, float, float] = (0.025, 0.5, 0.975)


def chronological_split(n_samples: int, train_ratio: float, val_ratio: float) -> tuple[slice, slice, slice]:
    """Return chronological train/validation/test slices."""
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n_samples)


def make_windows(values: Iterable[float], context_length: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct one-step or multi-step forecasting windows from a scalar series."""
    arr = np.asarray(list(values), dtype=np.float32)
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    last_start = len(arr) - context_length - horizon + 1
    for start in range(max(last_start, 0)):
        end = start + context_length
        target_idx = end + horizon - 1
        x_rows.append(arr[start:end])
        y_rows.append(float(arr[target_idx]))
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(100 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))


def interval_coverage_rate(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(100 * np.mean((y_true >= lower) & (y_true <= upper)))


def interval_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.05) -> float:
    width = upper - lower
    lower_penalty = (2 / alpha) * (lower - y_true) * (y_true < lower)
    upper_penalty = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return float(np.mean(width + lower_penalty + upper_penalty))


def evaluate_quantile_predictions(frame: pd.DataFrame) -> dict[str, float]:
    """Compute the public point and interval metrics from released prediction CSVs."""
    y_true = frame["true_value"].to_numpy(dtype=float)
    y_med = frame["pred_med"].to_numpy(dtype=float)
    y_low = frame["pred_low"].to_numpy(dtype=float)
    y_high = frame["pred_high"].to_numpy(dtype=float)
    return {
        "SMAPE": smape(y_true, y_med),
        "ICR": interval_coverage_rate(y_true, y_low, y_high),
        "IS": interval_score(y_true, y_low, y_high),
        "MAE": float(np.mean(np.abs(y_true - y_med))),
        "RMSE": float(np.sqrt(np.mean((y_true - y_med) ** 2))),
    }


def build_agru_pn_model_stub(config: TrainingConfig):
    """Placeholder for the unpublished AGruPn core model.

    The public notebooks provide partial implementations for interpolation,
    regime splitting, and quantile prediction experiments. The final integrated
    AGruPn training block is omitted because it is being extended in ongoing
    research. Users can connect their own GRU/PINN/quantile model here while
    keeping the same data protocol and metrics.
    """
    raise NotImplementedError(
        "The complete AGruPn core training module is not included in this public release."
    )


def tune_public_hyperparameters(config: TrainingConfig) -> dict[str, str]:
    """Document the tuning policy used in the manuscript."""
    return {
        "standard_forecasting_parameters": "grid search over context length, learning rate, batch size, hidden size, and layer count",
        "physics_and_loss_parameters": "particle swarm optimization for lambda, lambda1, lambda2, lambda3, and related loss weights",
        "selection_rule": "validation-set SMAPE, ICR, and interval score with chronological splits",
        "random_seed": str(config.random_seed),
    }


def main() -> None:
    config = TrainingConfig()
    print("AGruPn public training protocol")
    print(f"Release directory: {RELEASE_DIR}")
    print(f"Default prediction-data directory: {DEFAULT_DATA_DIR}")
    print(f"Training configuration: {config}")
    print("Tuning policy:")
    for key, value in tune_public_hyperparameters(config).items():
        print(f"  - {key}: {value}")
    print("\nThe integrated AGruPn core training block is intentionally omitted.")
    print("Use the released notebooks and metric scripts for the public reproducibility components.")


if __name__ == "__main__":
    main()
