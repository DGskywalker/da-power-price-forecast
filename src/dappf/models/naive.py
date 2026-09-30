"""Statistical and Heuristic Benchmarks for Day-Ahead Power Price Forecasting."""

from typing import Dict, Optional
import numpy as np
import pandas as pd


class Naive24hModel:
    """Naive-24h: price(t) = price(t - 24h). Predicts yesterday's price at the same hour."""

    def __init__(self, target_col: str = "price_eur_mwh"):
        self.target_col = target_col
        self.lag_col = f"{target_col}_lag_24"

    def fit(self, df_train: pd.DataFrame) -> "Naive24hModel":
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        if self.lag_col in df_test.columns:
            return df_test[self.lag_col].values
        # Fallback if raw target exists
        return df_test[self.target_col].shift(24).bfill().values


class NaiveWeekModel:
    """Naive-week: price(t) = price(t - 168h). Predicts same hour last week."""

    def __init__(self, target_col: str = "price_eur_mwh"):
        self.target_col = target_col
        self.lag_col = f"{target_col}_lag_168"

    def fit(self, df_train: pd.DataFrame) -> "NaiveWeekModel":
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        if self.lag_col in df_test.columns:
            return df_test[self.lag_col].values
        return df_test[self.target_col].shift(168).bfill().values


class SeasonalNaiveAverageModel:
    """
    Seasonal-naive-average: mean of same hour over the last 7 days:
    (1/7) * sum_{k=1}^7 price(t - 24 * k).
    """

    def __init__(self, target_col: str = "price_eur_mwh"):
        self.target_col = target_col

    def fit(self, df_train: pd.DataFrame) -> "SeasonalNaiveAverageModel":
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        # If lags 24, 48, 72 are in columns, average them or compute on rolling basis
        lags = [24, 48, 72]
        available_lags = [f"{self.target_col}_lag_{l}" for l in lags if f"{self.target_col}_lag_{l}" in df_test.columns]
        if available_lags:
            return df_test[available_lags].mean(axis=1).values
        # Rolling 168h mean shifted 24h as equivalent multi-day proxy
        roll_col = f"{self.target_col}_roll_mean_168h"
        if roll_col in df_test.columns:
            return df_test[roll_col].values
        return df_test[self.target_col].shift(24).bfill().values


class ClimatologicalMeanModel:
    """Climatological mean: historical mean price for that hour-of-week over the entire training set."""

    def __init__(self, target_col: str = "price_eur_mwh"):
        self.target_col = target_col
        self.hour_of_week_means: Dict[int, float] = {}
        self.global_mean: float = 0.0

    def fit(self, df_train: pd.DataFrame) -> "ClimatologicalMeanModel":
        train_df = df_train.copy()
        if "dayofweek" not in train_df.columns or "hour" not in train_df.columns:
            ts = pd.to_datetime(train_df["timestamp"], utc=True)
            train_df["dayofweek"] = ts.dt.dayofweek
            train_df["hour"] = ts.dt.hour

        train_df["how"] = train_df["dayofweek"] * 24 + train_df["hour"]
        self.hour_of_week_means = train_df.groupby("how")[self.target_col].mean().to_dict()
        self.global_mean = float(train_df[self.target_col].mean())
        return self

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        test_df = df_test.copy()
        if "dayofweek" not in test_df.columns or "hour" not in test_df.columns:
            ts = pd.to_datetime(test_df["timestamp"], utc=True)
            test_df["dayofweek"] = ts.dt.dayofweek
            test_df["hour"] = ts.dt.hour

        how = test_df["dayofweek"].values * 24 + test_df["hour"].values
        preds = np.array([self.hour_of_week_means.get(h, self.global_mean) for h in how])
        return preds
