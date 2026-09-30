"""Electricity Market Fundamentals: Residual Load, Renewable Penetration, Ramps, and Spreads.

Strictly adheres to Day-Ahead Gate Closure: any actual realization is shifted by >= 24h.
"""

from typing import Optional
import numpy as np
import pandas as pd


def add_fundamental_features(df: pd.DataFrame, zone: str = "DE_LU") -> pd.DataFrame:
    """
    Compute domain-specific electricity market fundamentals.

    Gate-closure safety guarantee:
    Any realized quantity (actual load, realized generation) is lagged by at least 24h
    before computing residual load, renewable shares, and ramp velocities.
    """
    df = df.copy()

    # Identify whether we have day-ahead forecasts (ex-ante) or realized actuals
    # Ex-ante forecasts available before gate closure:
    has_da_demand_forecast = "da_demand_forecast_mw" in df.columns

    # For realized generation/load, shift by 24h to ensure full gate-closure compliance
    for col in ["actual_load", "gen_wind", "gen_wind_onshore", "gen_wind_offshore", "gen_solar", "gen_gas", "gen_coal", "gen_nuclear"]:
        if col in df.columns and f"{col}_lag_24" not in df.columns:
            df[f"{col}_lag_24"] = df[col].shift(24)

    # Use ex-ante load forecast if present, otherwise use 24h-lagged load
    if has_da_demand_forecast:
        load_signal = df["da_demand_forecast_mw"]
    elif "actual_load_lag_24" in df.columns:
        load_signal = df["actual_load_lag_24"]
    else:
        load_signal = None

    # Wind and solar signals: use weather forecast proxies or 24h lagged generation
    if "wind_speed_cubed" in df.columns:
        # Physical proxy: kinetic power proportional to v^3
        wind_proxy = df["wind_speed_cubed"] * (1500.0 if zone == "DE_LU" else 900.0)
    elif "gen_wind_lag_24" in df.columns:
        wind_proxy = df["gen_wind_lag_24"]
    else:
        wind_proxy = 0.0

    if "solar_radiation" in df.columns:
        solar_proxy = df["solar_radiation"] * (50.0 if zone == "DE_LU" else 20.0)
    elif "gen_solar_lag_24" in df.columns:
        solar_proxy = df["gen_solar_lag_24"]
    else:
        solar_proxy = 0.0

    df["fundamental_wind_proxy"] = wind_proxy
    df["fundamental_solar_proxy"] = solar_proxy
    df["fundamental_vre_proxy"] = wind_proxy + solar_proxy

    if load_signal is not None:
        df["fundamental_load_signal"] = load_signal
        # 1. Residual Load Proxy
        df["residual_load_proxy"] = load_signal - df["fundamental_vre_proxy"]

        # 2. Renewable Penetration %
        eps = 1e-4
        df["renewable_penetration_ratio"] = np.clip(
            df["fundamental_vre_proxy"] / (load_signal + eps), 0.0, 2.5
        )

        # 3. Ramps (using gate-closure compliant signals)
        df["load_ramp_24h_diff"] = load_signal.diff(1)
        df["wind_ramp_24h_diff"] = pd.Series(wind_proxy).diff(1)
        df["solar_ramp_24h_diff"] = pd.Series(solar_proxy).diff(1)

        # 4. Scarcity stress metric (ratio to 168h rolling lagged residual load)
        roll_res = df["residual_load_proxy"].shift(24).rolling(window=168, min_periods=24).mean()
        df["residual_load_scarcity_ratio"] = df["residual_load_proxy"] / (roll_res + eps)

    # 5. Clean spark spread proxy using previous day's known price and fuel benchmarks
    price_col = "price_eur_mwh" if zone == "DE_LU" else "price_gbp_mwh"
    if f"{price_col}_lag_24" in df.columns:
        gas_proxy = 35.0 if zone == "DE_LU" else 30.0
        carbon_proxy = 70.0 if zone == "DE_LU" else 55.0
        srmc_proxy = 2.0 * gas_proxy + 0.38 * carbon_proxy
        df["spark_spread_lag_24"] = df[f"{price_col}_lag_24"] - srmc_proxy

    return df
