"""Leakage-safe Lag, Rolling, and Volatility Features respecting Day-Ahead Gate Closure."""

from typing import List, Optional
import numpy as np
import pandas as pd


def add_leakage_safe_lags(
    df: pd.DataFrame,
    price_col: str = "price_eur_mwh",
    gate_closure_shift: int = 24
) -> pd.DataFrame:
    """
    Construct strictly leakage-safe lag and rolling features.

    All features constructed for target hour t are strictly shifted by at least
    gate_closure_shift hours (default: 24h), ensuring they were determined and
    published before the Day-Ahead auction gate closure (12:00 CET / 11:00 GMT).
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Direct autoregressive price lags (24h, 48h, 72h, 168h)
    for lag in [24, 48, 72, 168]:
        df[f"{price_col}_lag_{lag}"] = df[price_col].shift(lag)

    # 2. Rolling statistics shifted by 24h to avoid using future or concurrent values
    shifted_price = df[price_col].shift(gate_closure_shift)

    # 24h rolling window (daily statistics of known day D-1)
    df[f"{price_col}_roll_mean_24h"] = shifted_price.rolling(window=24, min_periods=12).mean()
    df[f"{price_col}_roll_std_24h"] = shifted_price.rolling(window=24, min_periods=12).std()
    df[f"{price_col}_roll_min_24h"] = shifted_price.rolling(window=24, min_periods=12).min()
    df[f"{price_col}_roll_max_24h"] = shifted_price.rolling(window=24, min_periods=12).max()

    # 168h rolling window (weekly statistics of known past week)
    df[f"{price_col}_roll_mean_168h"] = shifted_price.rolling(window=168, min_periods=48).mean()
    df[f"{price_col}_roll_std_168h"] = shifted_price.rolling(window=168, min_periods=48).std()
    df[f"{price_col}_roll_min_168h"] = shifted_price.rolling(window=168, min_periods=48).min()
    df[f"{price_col}_roll_max_168h"] = shifted_price.rolling(window=168, min_periods=48).max()

    # 3. Previous-day DA peak vs off-peak spread and daily volatility
    # Daily grouping: compute peak (hours 8-20) and off-peak (remaining) on day D-1
    hour_series = df["timestamp"].dt.hour
    is_peak = ((hour_series >= 8) & (hour_series <= 20)).astype(int)
    
    # Compute expanding/rolling peak and off-peak averages over the 24h shifted window
    peak_price_shifted = shifted_price.where(is_peak == 1)
    offpeak_price_shifted = shifted_price.where(is_peak == 0)

    # Rolling 24h forward-filled proxies for previous-day peak and off-peak
    roll_peak = peak_price_shifted.rolling(window=24, min_periods=4).mean()
    roll_offpeak = offpeak_price_shifted.rolling(window=24, min_periods=4).mean()

    df[f"{price_col}_prev_day_peak_offpeak_spread"] = (roll_peak - roll_offpeak).ffill()
    df[f"{price_col}_prev_day_volatility"] = df[f"{price_col}_roll_std_24h"].ffill()

    # 4. Lagged fundamental variables (load, wind, solar) shifted by 24h
    for col in ["actual_load", "gen_wind", "gen_solar", "gen_gas", "residual_load"]:
        if col in df.columns:
            df[f"{col}_lag_24"] = df[col].shift(24)
            df[f"{col}_lag_48"] = df[col].shift(48)
            df[f"{col}_roll_mean_24h"] = df[col].shift(24).rolling(window=24, min_periods=12).mean()

    return df
