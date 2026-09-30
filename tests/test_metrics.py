"""Unit tests for quantitative evaluation metrics and statistical loss functions."""

import numpy as np
import pytest
from dappf.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_smape,
    calculate_pinball_loss,
    calculate_interval_coverage,
    calculate_bias,
    compute_all_metrics,
)


def test_mae_rmse_bias_standard():
    y_true = np.array([100.0, 50.0, 80.0, 60.0])
    y_pred = np.array([110.0, 45.0, 80.0, 50.0])

    # Errors: [10, -5, 0, -10]
    # Abs errors: [10, 5, 0, 10] -> MAE = 25 / 4 = 6.25
    mae = calculate_mae(y_true, y_pred)
    assert np.isclose(mae, 6.25)

    # Squared errors: [100, 25, 0, 100] -> MSE = 225 / 4 = 56.25 -> RMSE = 7.5
    rmse = calculate_rmse(y_true, y_pred)
    assert np.isclose(rmse, 7.5)

    # Bias: mean(y_pred - y_true) = (10 - 5 + 0 - 10) / 4 = -1.25
    bias = calculate_bias(y_true, y_pred)
    assert np.isclose(bias, -1.25)


def test_smape_zero_handling():
    # Symmetric MAPE handles near-zero or negative prices gracefully
    y_true = np.array([0.0, 10.0, -5.0])
    y_pred = np.array([0.0, 12.0, -5.0])

    smape = calculate_smape(y_true, y_pred)
    assert np.isfinite(smape)
    assert smape >= 0.0


def test_pinball_loss():
    # Pinball loss: L_alpha(y, q) = max(alpha * (y - q), (alpha - 1) * (y - q))
    y_true = np.array([100.0])
    # Underprediction (y > q)
    q_pred_under = np.array([90.0])
    loss_09_under = calculate_pinball_loss(y_true, q_pred_under, alpha=0.9)
    assert np.isclose(loss_09_under, 0.9 * 10.0)

    # Overprediction (y < q)
    q_pred_over = np.array([110.0])
    loss_01_over = calculate_pinball_loss(y_true, q_pred_over, alpha=0.1)
    assert np.isclose(loss_01_over, (1.0 - 0.1) * 10.0)


def test_prediction_interval_coverage():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    lower = np.array([5.0, 15.0, 25.0, 35.0, 60.0])  # Last one misses
    upper = np.array([15.0, 25.0, 35.0, 45.0, 70.0])

    # 4 out of 5 covered -> 80% coverage
    cov = calculate_interval_coverage(y_true, lower, upper)
    assert np.isclose(cov, 0.8)
