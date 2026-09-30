"""Quantitative Evaluation Metrics for Electricity Price Forecasting (EPF)."""

from typing import Dict, Optional
import numpy as np
import pandas as pd


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error: (1/N) * sum(|y_true - y_pred|)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error: sqrt((1/N) * sum((y_true - y_pred)^2))."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).
    Robust to near-zero or negative electricity prices:
    sMAPE = (100 / N) * sum( 2 * |y_pred - y_true| / (|y_true| + |y_pred| + eps) )
    """
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denominator))


def calculate_pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    """
    Pinball Loss (Quantile Loss) for target quantile alpha in (0, 1):
    L_alpha(y, q) = max(alpha * (y - q), (alpha - 1) * (y - q))
    """
    err = y_true - q_pred
    loss = np.maximum(alpha * err, (alpha - 1.0) * err)
    return float(np.mean(loss))


def calculate_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> float:
    """Empirical coverage probability: fraction of actual prices inside [lower, upper]."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def calculate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Forecast Bias: (1/N) * sum(y_pred - y_true)."""
    return float(np.mean(y_pred - y_true))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_80: Optional[np.ndarray] = None,
    upper_80: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute complete quantitative performance suite."""
    metrics = {
        "mae": calculate_mae(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "smape": calculate_smape(y_true, y_pred),
        "bias": calculate_bias(y_true, y_pred),
    }

    if lower_80 is not None and upper_80 is not None:
        metrics["coverage_80"] = calculate_interval_coverage(y_true, lower_80, upper_80)
        metrics["pinball_10"] = calculate_pinball_loss(y_true, lower_80, alpha=0.1)
        metrics["pinball_90"] = calculate_pinball_loss(y_true, upper_80, alpha=0.9)
        metrics["avg_pinball"] = 0.5 * (metrics["pinball_10"] + metrics["pinball_90"])

    return metrics
