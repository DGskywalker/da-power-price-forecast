"""Feature Engineering Orchestrator: Assembles Clean Tidy Parquet Dataset."""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger
from dappf.config import AppConfig, load_config
from dappf.features.calendar import add_calendar_features
from dappf.features.fundamentals import add_fundamental_features
from dappf.features.lags import add_leakage_safe_lags

RAW_SIMULTANEOUS_COLUMNS = [
    "actual_load",
    "gen_wind",
    "gen_wind_onshore",
    "gen_wind_offshore",
    "gen_solar",
    "gen_gas",
    "gen_coal",
    "gen_nuclear",
    "gen_lignite",
    "gen_biomass",
    "residual_load",
    "cross_border_flow",
    "residual_load_reported",
    "volume_mwh",
    "interconnector_flow",
]


def assemble_features(df: pd.DataFrame, zone: str = "DE_LU") -> pd.DataFrame:
    """Run full feature engineering and drop simultaneous unlagged actuals."""
    target_col = "price_eur_mwh" if zone == "DE_LU" else "price_gbp_mwh"

    # 1. Leakage-safe Lags and Rolling Windows (shifted >= 24h)
    df = add_leakage_safe_lags(df, price_col=target_col, gate_closure_shift=24)

    # 2. Fundamentals (strict gate-closure compliance)
    df = add_fundamental_features(df, zone=zone)

    # 3. Calendar, DST, Holidays, and Fourier Harmonics
    df = add_calendar_features(df, zone=zone)

    # 4. Drop raw simultaneous realization columns to strictly prevent lookahead
    drop_cols = [c for c in RAW_SIMULTANEOUS_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


class FeatureBuilder:
    """Builds clean, leakage-safe feature matrices for DE-LU and GB bidding zones."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.cfg = config or load_config()
        self.raw_dir = Path(self.cfg.paths.raw_dir)
        self.processed_dir = Path(self.cfg.paths.processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_zone_raw(self, zone: str) -> pd.DataFrame:
        """Load and merge raw market and weather parquet files for zone."""
        market_dir = self.raw_dir / "market" / zone
        weather_dir = self.raw_dir / "weather" / zone

        market_files = sorted(market_dir.glob("*.parquet")) if market_dir.exists() else []
        weather_files = sorted(weather_dir.glob("*.parquet")) if weather_dir.exists() else []

        if not market_files:
            raise FileNotFoundError(f"No raw market data found for {zone} in {market_dir}")
        if not weather_files:
            raise FileNotFoundError(f"No raw weather data found for {zone} in {weather_dir}")

        df_m = pd.concat([pd.read_parquet(f) for f in market_files], ignore_index=True)
        df_w = pd.concat([pd.read_parquet(f) for f in weather_files], ignore_index=True)

        df_m["timestamp"] = pd.to_datetime(df_m["timestamp"], utc=True)
        df_w["timestamp"] = pd.to_datetime(df_w["timestamp"], utc=True)

        df_m = df_m.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df_w = df_w.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        # Merge on UTC timestamp
        merged = pd.merge(df_m, df_w, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
        return merged

    def build_features_for_zone(self, zone: str) -> pd.DataFrame:
        """Construct full feature pipeline for a zone."""
        logger.info(f"Building features for bidding zone: {zone}...")
        raw_df = self.load_zone_raw(zone)
        target_col = "price_eur_mwh" if zone == "DE_LU" else "price_gbp_mwh"

        df = assemble_features(raw_df, zone=zone)

        # Interpolate brief missing target periods (e.g. DST hour transitions)
        df[target_col] = df[target_col].interpolate(method="linear").bfill().ffill()

        # Drop initial rows that have NaNs due to the 168h (7-day) lag window
        init_len = len(df)
        df = df.dropna(subset=[f"{target_col}_lag_168", f"{target_col}_roll_mean_168h"]).reset_index(drop=True)
        # Fill any remaining feature NaNs with forward/backward fill
        df = df.bfill().ffill().fillna(0.0)
        logger.info(f"[{zone}] Dropped {init_len - len(df)} warm-up rows for 168h lag. Clean rows: {len(df)}")

        # Save to processed parquet
        out_file = self.processed_dir / f"features_{zone}.parquet"
        df.to_parquet(out_file, index=False)
        logger.info(f"[{zone}] Features saved to {out_file} (shape: {df.shape})")

        return df

    def build_all(self) -> Dict[str, pd.DataFrame]:
        """Build features for all bidding zones."""
        results = {}
        for zone in self.cfg.zones.keys():
            results[zone] = self.build_features_for_zone(zone)
        return results
