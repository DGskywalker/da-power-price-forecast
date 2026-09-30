"""Hybrid Model: PyTorch Multi-Horizon LSTM + Residual LightGBM Booster."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger
from dappf.models.gbdt import LightGBMForecaster
from dappf.models.lstm import LSTMForecaster


class HybridForecaster:
    """
    Two-stage Hybrid Forecaster:
    Stage 1: Multi-Horizon LSTM predicts the underlying non-linear sequence pattern.
    Stage 2: LightGBM predicts the residual errors using market fundamentals and renewable ramps.
    """

    def __init__(
        self,
        lstm_params: Optional[dict] = None,
        lgb_params: Optional[dict] = None
    ):
        self.lstm = LSTMForecaster(**(lstm_params or {}))
        self.lgb_residual = LightGBMForecaster(params=lgb_params)
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "HybridForecaster":
        """Fit LSTM on full feature matrix, compute residuals on train set, and train LightGBM on residuals."""
        logger.info("Fitting Stage 1: PyTorch LSTM...")
        self.feature_names = list(X_train.columns)
        self.lstm.fit(X_train, y_train)

        # Generate in-sample LSTM predictions (using 24h rolling block inference)
        logger.info("Computing Stage 1 training residuals...")
        # To avoid data leakage, predict residuals starting after the lookback sequence
        start_idx = self.lstm.seq_len
        lstm_train_preds = self.lstm.predict(X_train, test_start_idx=start_idx)

        y_eval = y_train.iloc[start_idx:].values
        residuals = y_eval - lstm_train_preds

        # Stage 2: Fit LightGBM on the residual errors
        logger.info("Fitting Stage 2: LightGBM on residual error signal...")
        X_res_train = X_train.iloc[start_idx:].reset_index(drop=True)
        y_res_train = pd.Series(residuals, index=X_res_train.index)

        self.lgb_residual.fit(X_res_train, y_res_train, fit_quantiles=False)
        return self

    def predict(
        self,
        X_full: pd.DataFrame,
        test_start_idx: int,
    ) -> np.ndarray:
        """
        Produce combined prediction:
        y_hybrid = y_lstm + residual_lgb
        """
        # Stage 1: LSTM base forecast
        lstm_preds = self.lstm.predict(X_full, test_start_idx=test_start_idx)

        # Stage 2: LightGBM residual forecast
        X_test = X_full.iloc[test_start_idx:].reset_index(drop=True)
        residual_preds = self.lgb_residual.predict(X_test)

        final_preds = lstm_preds + residual_preds
        return final_preds
