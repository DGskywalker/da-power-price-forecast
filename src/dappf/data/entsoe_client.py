"""ENTSO-E Transparency Platform and SMARD/Energy-Charts Data Ingestion Client."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import os
import json
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
from loguru import logger

try:
    from entsoe import EntsoePandasClient
except ImportError:
    EntsoePandasClient = None


class EntsoeClient:
    """Client for retrieving European day-ahead electricity prices, load, and generation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ENTSOE_API_KEY")
        self.py_client = None
        if self.api_key and EntsoePandasClient is not None:
            try:
                self.py_client = EntsoePandasClient(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize EntsoePandasClient: {e}")

    def fetch_de_energy_charts(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch DE-LU day-ahead prices, generation, and load from Energy-Charts API."""
        logger.info(f"Fetching DE-LU data from Energy-Charts API ({start_date} to {end_date})...")
        
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)
        
        # Energy-charts allows fetching year by year or range
        dfs = []
        for year in range(start_dt.year, end_dt.year + 1):
            y_start = f"{year}-01-01T00:00+01:00"
            y_end = f"{year}-12-31T23:00+01:00"
            
            # 1. Fetch Prices
            price_url = f"https://api.energy-charts.info/price?bzn=DE-LU&start={urllib.parse.quote(y_start)}&end={urllib.parse.quote(y_end)}"
            try:
                req = urllib.request.Request(price_url, headers={"User-Agent": "dappf-research/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    p_data = json.loads(resp.read().decode())
                
                df_p = pd.DataFrame({
                    "timestamp": pd.to_datetime(p_data["unix_seconds"], unit="s", utc=True),
                    "price_eur_mwh": p_data["price"]
                }).dropna(subset=["timestamp"])
            except Exception as e:
                logger.warning(f"Energy-charts price fetch failed for year {year}: {e}")
                df_p = pd.DataFrame(columns=["timestamp", "price_eur_mwh"])

            # 2. Fetch Public Power (Generation & Load)
            power_url = f"https://api.energy-charts.info/public_power?bzn=DE-LU&start={urllib.parse.quote(y_start)}&end={urllib.parse.quote(y_end)}"
            try:
                req = urllib.request.Request(power_url, headers={"User-Agent": "dappf-research/1.0"})
                with urllib.request.urlopen(req, timeout=40) as resp:
                    pw_data = json.loads(resp.read().decode())
                
                pw_ts = pd.to_datetime(pw_data["unix_seconds"], unit="s", utc=True)
                pw_dict = {"timestamp": pw_ts}
                
                for p_type in pw_data.get("production_types", []):
                    name = p_type["name"].lower()
                    data = p_type["data"]
                    if "wind onshore" in name:
                        pw_dict["gen_wind_onshore"] = data
                    elif "wind offshore" in name:
                        pw_dict["gen_wind_offshore"] = data
                    elif "solar" in name:
                        pw_dict["gen_solar"] = data
                    elif "fossil gas" in name:
                        pw_dict["gen_gas"] = data
                    elif "fossil hard coal" in name:
                        pw_dict["gen_coal"] = data
                    elif "fossil brown coal" in name:
                        pw_dict["gen_lignite"] = data
                    elif "nuclear" in name:
                        pw_dict["gen_nuclear"] = data
                    elif "biomass" in name:
                        pw_dict["gen_biomass"] = data
                    elif "load" == name:
                        pw_dict["actual_load"] = data
                    elif "residual load" in name:
                        pw_dict["residual_load_reported"] = data
                    elif "cross border" in name:
                        pw_dict["cross_border_flow"] = data
                
                df_pw = pd.DataFrame(pw_dict).dropna(subset=["timestamp"])
                # Resample 15-min to hourly mean
                df_pw = df_pw.set_index("timestamp").resample("1h").mean().reset_index()
            except Exception as e:
                logger.warning(f"Energy-charts power fetch failed for year {year}: {e}")
                df_pw = pd.DataFrame(columns=["timestamp"])

            # Merge price and power for this year
            if not df_p.empty and not df_pw.empty:
                df_y = pd.merge(df_p, df_pw, on="timestamp", how="outer")
            elif not df_p.empty:
                df_y = df_p
            else:
                df_y = df_pw
            
            if not df_y.empty:
                dfs.append(df_y)

        if not dfs:
            raise RuntimeError("Failed to fetch DE-LU data from Energy-Charts")
            
        full_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["timestamp"])
        full_df = full_df.sort_values("timestamp").reset_index(drop=True)
        # Filter to requested range
        full_df = full_df[(full_df["timestamp"] >= start_dt) & (full_df["timestamp"] <= end_dt)]
        return full_df

    def fetch_or_fallback(self, zone: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch zone data via ENTSO-E API or fall back to open mirror."""
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC")

        if self.py_client and zone == "DE_LU":
            try:
                logger.info(f"Attempting ENTSO-E direct fetch for {zone}...")
                country_code = "DE_LU"
                prices = self.py_client.query_day_ahead_prices(country_code, start=start_ts, end=end_ts)
                load = self.py_client.query_load(country_code, start=start_ts, end=end_ts)
                gen = self.py_client.query_generation(country_code, start=start_ts, end=end_ts)
                df = pd.DataFrame({"price_eur_mwh": prices, "actual_load": load})
                df = df.join(gen, how="outer").resample("1h").mean().reset_index()
                df.rename(columns={"index": "timestamp"}, inplace=True)
                return df
            except Exception as e:
                logger.warning(f"Direct ENTSO-E fetch failed: {e}. Falling back to open mirror.")

        if zone == "DE_LU":
            return self.fetch_de_energy_charts(start_date, end_date)
        else:
            raise NotImplementedError(f"Direct mirror for {zone} not in entsoe_client; use Elexon for GB")
