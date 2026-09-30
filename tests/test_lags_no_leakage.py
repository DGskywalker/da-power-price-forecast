"""Programmatic Verification: Strict Day-Ahead Gate-Closure Leakage Prohibition."""

import numpy as np
import pandas as pd
import pytest
from dappf.features.build import assemble_features


def test_leakage_safe_lags_programmatic_perturbation():
    """
    Assert that features for hour t are strictly computable using only information
    available prior to day-ahead gate closure (shifted by >= 24h).

    Verification protocol:
    1. Sample at least 5 random forecast target timestamps t.
    2. Perturb all raw market prices and fundamentals for all timestamps tau > (t - 24h).
    3. Recompute all engineered features.
    4. Assert that for the target timestamp t, feature values are 100% IDENTICAL to the baseline.
    """
    np.random.seed(42)
    n_hours = 600  # 25 days of hourly data
    dates = pd.date_range("2023-01-01 00:00:00", periods=n_hours, freq="1h", tz="UTC")

    base_price = 50.0 + 20.0 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + np.random.normal(0, 5, n_hours)
    base_load = 40000.0 + 10000.0 * np.sin(np.arange(n_hours) * 2 * np.pi / 24)
    base_wind = np.maximum(0.0, 15000.0 + np.random.normal(0, 3000, n_hours))
    base_solar = np.maximum(0.0, 10000.0 * np.sin(np.arange(n_hours) * np.pi / 12))

    df_clean = pd.DataFrame({
        "timestamp": dates,
        "price_eur_mwh": base_price,
        "actual_load": base_load,
        "gen_wind": base_wind,
        "gen_solar": base_solar,
    })

    # Run baseline feature pipeline
    feats_baseline = assemble_features(df_clean, zone="DE_LU")

    # Sample at least 5 target hours well after the 168h warm-up period
    sample_indices = [200, 255, 312, 420, 500, 550]
    feature_cols = [c for c in feats_baseline.columns if c not in ["timestamp", "price_eur_mwh"]]

    for idx in sample_indices:
        target_timestamp = df_clean.iloc[idx]["timestamp"]
        # Gate closure cutoff for target hour t: any information after (target_timestamp - 24h)
        # must NOT affect the features engineered at target_timestamp.
        cutoff_timestamp = target_timestamp - pd.Timedelta(hours=24)

        # Create corrupted dataframe where all post-cutoff data is heavily perturbed
        df_corrupted = df_clean.copy()
        post_cutoff_mask = df_corrupted["timestamp"] > cutoff_timestamp

        # Inject severe shocks into prices and fundamentals occurring AFTER gate closure
        df_corrupted.loc[post_cutoff_mask, "price_eur_mwh"] += 9999.0
        df_corrupted.loc[post_cutoff_mask, "actual_load"] += 88888.0
        df_corrupted.loc[post_cutoff_mask, "gen_wind"] = 0.0
        df_corrupted.loc[post_cutoff_mask, "gen_solar"] = 0.0

        # Recompute features on corrupted dataset
        feats_perturbed = assemble_features(df_corrupted, zone="DE_LU")

        # Compare feature row for target_timestamp
        row_base = feats_baseline.loc[feats_baseline["timestamp"] == target_timestamp, feature_cols].iloc[0]
        row_pert = feats_perturbed.loc[feats_perturbed["timestamp"] == target_timestamp, feature_cols].iloc[0]

        # Assert no features changed
        for col in feature_cols:
            val_base = row_base[col]
            val_pert = row_pert[col]
            if pd.isna(val_base) and pd.isna(val_pert):
                continue
            assert np.isclose(val_base, val_pert, rtol=1e-7, atol=1e-7), (
                f"LEAKAGE DETECTED in feature '{col}' at timestamp {target_timestamp}!\n"
                f"Baseline: {val_base}, Perturbed: {val_pert}\n"
                f"Perturbing raw data after gate closure {cutoff_timestamp} altered feature at {target_timestamp}!"
            )


def test_target_not_in_features():
    """Verify that current price target is never inadvertently included as a predictor."""
    dates = pd.date_range("2023-01-01 00:00:00", periods=200, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "price_eur_mwh": np.random.randn(200),
        "actual_load": 30000 + np.random.randn(200) * 1000,
        "gen_wind": 5000 + np.random.randn(200) * 500,
        "gen_solar": 2000 + np.random.randn(200) * 200,
    })
    res = assemble_features(df, zone="DE_LU")
    # All price features must explicitly have 'lag', 'roll', 'prev_day', or 'spark_spread' in their names
    price_cols = [c for c in res.columns if "price_eur_mwh" in c]
    for c in price_cols:
        if c == "price_eur_mwh":
            continue  # Target column
        assert any(k in c for k in ["lag", "roll", "prev_day", "spark_spread"]), f"Unsafe price feature: {c}"
