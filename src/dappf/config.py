from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import yaml
from pydantic import BaseModel, Field


class WeatherCity(BaseModel):
    name: str
    lat: float
    lon: float
    weight: float


class ZoneConfig(BaseModel):
    currency: str
    unit: str
    timezone: str
    entsoe_bzn: str
    gate_closure_cet: Optional[str] = "12:00"
    gate_closure_gmt: Optional[str] = "11:00"
    energy_charts_bzn: Optional[str] = None
    elexon_market: Optional[str] = None
    weather_cities: List[WeatherCity] = Field(default_factory=list)


class DatesConfig(BaseModel):
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    test_months: int = 6


class PathsConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "artifacts/models"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"


class BacktestConfig(BaseModel):
    min_train_months: int = 18
    step_months: int = 1
    horizon_hours: int = 24
    refit_cadence: str = "monthly"


class AppConfig(BaseModel):
    seed: int = 42
    zones: Dict[str, ZoneConfig]
    dates: DatesConfig = Field(default_factory=DatesConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    models: Dict[str, Any] = Field(default_factory=dict)
    entsoe_api_key: Optional[str] = None
    elexon_api_key: Optional[str] = None


def load_config(config_dir: str | Path = "config") -> AppConfig:
    """Load and validate system configuration from YAML and environment variables."""
    cfg_dir = Path(config_dir)
    base_file = cfg_dir / "base.yaml"
    models_file = cfg_dir / "models.yaml"

    if not base_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {base_file}")

    with open(base_file, "r") as f:
        base_data = yaml.safe_load(f) or {}

    models_data = {}
    if models_file.exists():
        with open(models_file, "r") as f:
            models_data = yaml.safe_load(f) or {}

    base_data["models"] = models_data
    base_data["entsoe_api_key"] = os.getenv("ENTSOE_API_KEY")
    base_data["elexon_api_key"] = os.getenv("ELEXON_API_KEY")

    if os.getenv("SEED"):
        base_data["seed"] = int(os.getenv("SEED"))

    return AppConfig(**base_data)
