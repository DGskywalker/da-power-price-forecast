"""Open-Meteo Historical Weather Reanalysis Client for DE-LU and GB."""

from typing import Dict, List, Optional
import json
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
from loguru import logger
from dappf.config import WeatherCity


class WeatherClient:
    """Client for fetching historical ERA5 reanalysis and weather forecasts from Open-Meteo."""

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self):
        pass

    def fetch_city_weather(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """Fetch hourly weather variables for a given latitude and longitude."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,wind_speed_100m,shortwave_radiation,cloud_cover,precipitation",
            "timezone": "UTC"
        }
        url = f"{self.ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": "dappf-research/1.0"}

        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=35) as resp:
                    data = json.loads(resp.read().decode())
                
                hourly = data.get("hourly", {})
                if not hourly or "time" not in hourly:
                    raise ValueError(f"No hourly weather data in response for lat={lat}, lon={lon}")

                df = pd.DataFrame({
                    "timestamp": pd.to_datetime(hourly["time"], utc=True),
                    "temp_2m": hourly["temperature_2m"],
                    "wind_speed_100m": hourly["wind_speed_100m"],
                    "solar_radiation": hourly["shortwave_radiation"],
                    "cloud_cover": hourly["cloud_cover"],
                    "precipitation": hourly["precipitation"],
                })
                return df
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Weather query failed for ({lat}, {lon}): {e}")
                    raise
                logger.warning(f"Weather retry {attempt+1}/{max_retries} for ({lat}, {lon}): {e}")
        return pd.DataFrame()

    def fetch_regional_weather(
        self,
        cities: List[WeatherCity],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch weather across multiple cities and compute population-weighted regional proxies."""
        if not cities:
            raise ValueError("No weather cities provided")

        total_weight = sum(c.weight for c in cities)
        city_dfs = []

        for city in cities:
            logger.info(f"Fetching weather for {city.name} (weight: {city.weight / total_weight:.2%})...")
            df_city = self.fetch_city_weather(city.lat, city.lon, start_date, end_date)
            if df_city.empty:
                continue
            norm_weight = city.weight / total_weight
            df_city["norm_weight"] = norm_weight
            city_dfs.append(df_city)

        if not city_dfs:
            raise RuntimeError("Failed to fetch weather for all cities")

        # Combine by weighted average
        base_df = city_dfs[0][["timestamp"]].copy()
        
        for col in ["temp_2m", "wind_speed_100m", "solar_radiation", "cloud_cover", "precipitation"]:
            weighted_sum = np.zeros(len(base_df))
            for c_df in city_dfs:
                weighted_sum += c_df[col].values * c_df["norm_weight"].values[0]
            base_df[col] = weighted_sum

        # Derived energy proxies
        # Wind power generation is proportional to cubic wind speed (v^3)
        base_df["wind_speed_cubed"] = (base_df["wind_speed_100m"] / 10.0) ** 3
        # Heating degree days proxy (base 15.5 C) and cooling degree days (base 18.3 C)
        base_df["hdd_proxy"] = np.maximum(0.0, 15.5 - base_df["temp_2m"])
        base_df["cdd_proxy"] = np.maximum(0.0, base_df["temp_2m"] - 18.3)

        return base_df.sort_values("timestamp").reset_index(drop=True)
