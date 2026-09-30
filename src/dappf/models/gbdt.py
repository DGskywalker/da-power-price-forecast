"""Gradient Boosted Decision Tree (GBDT) Forecasters: LightGBM, XGBoost, and CatBoost."""

from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
import optuna
from loguru import logger

optuna.logging.set_verbosity(optuna.logging.WARNING)


class LightGBMForecaster:
    """LightGBM Multi-Horizon Price Forecaster with Quantile Intervals."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        quantiles: Optional[List[float]] = None
    ):
        self.params = params or {
            "objective": "regression_l1",
            "metric": "mae",
            "boosting_type": "gbdt",
            "n_estimators": 250,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.models: Dict[int, lgb.LGBMRegressor] = {}
        self.quantile_models: Dict[float, Dict[int, lgb.LGBMRegressor]] = {q: {} for q in self.quantiles}
        self.feature_names: List[str] = []

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 15,
        timeout: int = 180
    ) -> Dict[str, Any]:
        """Tune key hyperparameters using Optuna TPE."""
        logger.info(f"Starting Optuna hyperparameter optimization ({n_trials} trials)...")

        def objective(trial: optuna.Trial) -> float:
            param = {
                "objective": "regression_l1",
                "metric": "mae",
                "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
                "n_estimators": 180,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            model = lgb.LGBMRegressor(**param)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return float(np.mean(np.abs(y_val - preds)))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        logger.info(f"Best Optuna MAE: {study.best_value:.3f} with params: {study.best_params}")
        self.params.update(study.best_params)
        return self.params

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hours_train: Optional[pd.Series] = None,
        fit_quantiles: bool = True
    ) -> "LightGBMForecaster":
        """Fit specialized models per hour of day (direct multi-horizon)."""
        self.feature_names = list(X_train.columns)
        
        if hours_train is None and "hour" in X_train.columns:
            hours = X_train["hour"]
        elif hours_train is not None:
            hours = hours_train
        else:
            hours = pd.Series(0, index=X_train.index)

        # Train a model for each delivery hour h in 0..23
        for h in range(24):
            mask = (hours == h) & (~y_train.isna()) & np.isfinite(y_train)
            if not mask.any():
                continue
            X_h = X_train.loc[mask]
            y_h = y_train.loc[mask]

            # Primary point forecast model
            m = lgb.LGBMRegressor(**self.params)
            m.fit(X_h, y_h)
            self.models[h] = m

            # Quantile models for prediction intervals
            if fit_quantiles:
                for q in self.quantiles:
                    q_params = dict(self.params)
                    q_params["objective"] = "quantile"
                    q_params["alpha"] = q
                    q_m = lgb.LGBMRegressor(**q_params)
                    q_m.fit(X_h, y_h)
                    self.quantile_models[q][h] = q_m

        return self

    def predict(
        self,
        X_test: pd.DataFrame,
        hours_test: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Predict point prices across test set."""
        if hours_test is None and "hour" in X_test.columns:
            hours = X_test["hour"]
        elif hours_test is not None:
            hours = hours_test
        else:
            hours = pd.Series(0, index=X_test.index)

        preds = np.zeros(len(X_test))
        for h, m in self.models.items():
            mask = (hours == h)
            if mask.any():
                preds[mask] = m.predict(X_test.loc[mask])
        return preds

    def predict_quantiles(
        self,
        X_test: pd.DataFrame,
        hours_test: Optional[pd.Series] = None
    ) -> Dict[float, np.ndarray]:
        """Predict quantile intervals for test set."""
        if hours_test is None and "hour" in X_test.columns:
            hours = X_test["hour"]
        elif hours_test is not None:
            hours = hours_test
        else:
            hours = pd.Series(0, index=X_test.index)

        q_preds = {}
        for q in self.quantiles:
            arr = np.zeros(len(X_test))
            for h, m in self.quantile_models[q].items():
                mask = (hours == h)
                if mask.any():
                    arr[mask] = m.predict(X_test.loc[mask])
            q_preds[q] = arr
        return q_preds


class XGBoostForecaster:
    """XGBoost Multi-Horizon Challenger Forecaster."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.models: Dict[int, xgb.XGBRegressor] = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostForecaster":
        hours = X_train["hour"] if "hour" in X_train.columns else pd.Series(0, index=X_train.index)
        for h in range(24):
            mask = (hours == h) & (~y_train.isna()) & np.isfinite(y_train)
            if not mask.any():
                continue
            m = xgb.XGBRegressor(**self.params)
            m.fit(X_train.loc[mask], y_train.loc[mask])
            self.models[h] = m
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        hours = X_test["hour"] if "hour" in X_test.columns else pd.Series(0, index=X_test.index)
        preds = np.zeros(len(X_test))
        for h, m in self.models.items():
            mask = (hours == h)
            if mask.any():
                preds[mask] = m.predict(X_test.loc[mask])
        return preds


class CatBoostForecaster:
    """CatBoost Multi-Horizon Challenger Forecaster."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "iterations": 200,
            "learning_rate": 0.05,
            "depth": 5,
            "random_seed": 42,
            "verbose": 0,
        }
        self.models: Dict[int, CatBoostRegressor] = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CatBoostForecaster":
        hours = X_train["hour"] if "hour" in X_train.columns else pd.Series(0, index=X_train.index)
        for h in range(24):
            mask = (hours == h) & (~y_train.isna()) & np.isfinite(y_train)
            if not mask.any():
                continue
            m = CatBoostRegressor(**self.params)
            m.fit(X_train.loc[mask], y_train.loc[mask])
            self.models[h] = m
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        hours = X_test["hour"] if "hour" in X_test.columns else pd.Series(0, index=X_test.index)
        preds = np.zeros(len(X_test))
        for h, m in self.models.items():
            mask = (hours == h)
            if mask.any():
                preds[mask] = m.predict(X_test.loc[mask])
        return preds
