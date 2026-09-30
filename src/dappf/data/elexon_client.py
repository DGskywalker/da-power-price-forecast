"""Elexon BMRS Insights Data Ingestion Client for Great Britain (GB)."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import json
import time
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
from loguru import logger


class ElexonClient:
    """Client for retrieving GB electricity market prices, demand, and generation from Elexon BMRS."""

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ELEXON_API_KEY")

    def _get_json(self, endpoint: str, params: Dict[str, str], max_retries: int = 3) -> dict:
        """Fetch JSON data with retry logic."""
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.BASE_URL}/{endpoint}?{query}"
        headers = {"User-Agent": "dappf-research/1.0", "Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode())
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"Request failed for {url}: {e}")
                    raise
                time.sleep(1.0 * (attempt + 1))
        return {}

    def fetch_market_index_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch GB Day-Ahead Market Index Data (APX & N2EX) half-hourly prices."""
        logger.info(f"Fetching GB Market Index Data (MID) prices ({start_date} to {end_date})...")
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        records = []
        curr = start_dt
        # Elexon MID enforces maximum 7-day window per request
        while curr < end_dt:
            nxt = min(curr + timedelta(days=6), end_dt)
            params = {
                "from": curr.strftime("%Y-%m-%d"),
                "to": nxt.strftime("%Y-%m-%d"),
                "format": "json"
            }
            try:
                res = self._get_json("datasets/MID", params)
                data = res.get("data", [])
                for row in data:
                    # Filter for APX or N2EX with non-zero volume/price
                    if row.get("price") is not None and row.get("dataProvider") == "APXMIDP":
                        records.append({
                            "timestamp": row.get("startTime"),
                            "price_gbp_mwh": float(row.get("price")),
                            "volume_mwh": float(row.get("volume", 0.0))
                        })
            except Exception as e:
                logger.warning(f"Error fetching MID between {curr} and {nxt}: {e}")
            curr = nxt

        if not records:
            logger.warning("No MID records retrieved directly; attempting fallback estimation")
            return pd.DataFrame(columns=["timestamp", "price_gbp_mwh"])

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # Aggregate half-hourly to hourly mean
        df = df.set_index("timestamp").resample("1h").mean().reset_index()
        return df

    def fetch_demand_forecast(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch National Demand Forecast (NDF) day-ahead half-hourly demand."""
        logger.info(f"Fetching GB Demand Forecasts ({start_date} to {end_date})...")
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        records = []
        curr = start_dt
        while curr < end_dt:
            nxt = min(curr + timedelta(days=30), end_dt)
            params = {
                "from": curr.strftime("%Y-%m-%d"),
                "to": nxt.strftime("%Y-%m-%d"),
                "format": "json"
            }
            try:
                res = self._get_json("datasets/NDF", params)
                for row in res.get("data", []):
                    if "demand" in row and "startTime" in row:
                        records.append({
                            "timestamp": row["startTime"],
                            "da_demand_forecast_mw": float(row["demand"])
                        })
            except Exception as e:
                logger.warning(f"Error fetching NDF between {curr} and {nxt}: {e}")
            curr = nxt

        if not records:
            return pd.DataFrame(columns=["timestamp", "da_demand_forecast_mw"])

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").resample("1h").mean().reset_index()
        return df

    def fetch_generation_by_type(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch GB actual generation by fuel type."""
        logger.info(f"Fetching GB generation by fuel type ({start_date} to {end_date})...")
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        records = []
        curr = start_dt
        while curr < end_dt:
            nxt = min(curr + timedelta(days=15), end_dt)
            params = {
                "from": curr.strftime("%Y-%m-%dT00:00:00Z"),
                "to": nxt.strftime("%Y-%m-%dT23:59:59Z"),
                "format": "json"
            }
            try:
                res = self._get_json("generation/actual/per-type", params)
                for item in res.get("data", []):
                    st = item.get("startTime")
                    row_dict = {"timestamp": st}
                    for g in item.get("data", []):
                        psr = g.get("psrType", "").lower()
                        qty = float(g.get("quantity", 0.0))
                        if "wind" in psr:
                            row_dict["gen_wind"] = row_dict.get("gen_wind", 0.0) + qty
                        elif "gas" in psr:
                            row_dict["gen_gas"] = row_dict.get("gen_gas", 0.0) + qty
                        elif "nuclear" in psr:
                            row_dict["gen_nuclear"] = row_dict.get("gen_nuclear", 0.0) + qty
                        elif "coal" in psr:
                            row_dict["gen_coal"] = row_dict.get("gen_coal", 0.0) + qty
                        elif "biomass" in psr:
                            row_dict["gen_biomass"] = row_dict.get("gen_biomass", 0.0) + qty
                        elif "solar" in psr:
                            row_dict["gen_solar"] = row_dict.get("gen_solar", 0.0) + qty
                        elif "interconnector" in psr:
                            row_dict["interconnector_flow"] = row_dict.get("interconnector_flow", 0.0) + qty
                    records.append(row_dict)
            except Exception as e:
                logger.warning(f"Error fetching generation between {curr} and {nxt}: {e}")
            curr = nxt

        if not records:
            return pd.DataFrame(columns=["timestamp"])

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").resample("1h").mean().reset_index()
        return df

    def fetch_gb_combined(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch and align all GB market data (price, demand, generation) into a unified dataframe."""
        df_p = self.fetch_market_index_prices(start_date, end_date)
        df_d = self.fetch_demand_forecast(start_date, end_date)
        df_g = self.fetch_generation_by_type(start_date, end_date)

        dfs = [d for d in [df_p, df_d, df_g] if not d.empty]
        if not dfs:
            raise RuntimeError("No GB market data could be fetched from Elexon.")

        combined = dfs[0]
        for nxt in dfs[1:]:
            combined = pd.merge(combined, nxt, on="timestamp", how="outer")

        combined = combined.sort_values("timestamp").reset_index(drop=True)
        return combined
