from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


def _env_interpolate(value: Any) -> Any:
    # Expand ${VAR:default} patterns using environment
    if isinstance(value, str):
        out = ""
        i = 0
        while i < len(value):
            if value[i:i+2] == "${":
                j = value.find("}", i+2)
                if j == -1:
                    out += value[i:]
                    break
                expr = value[i+2:j]
                if ":" in expr:
                    var, default = expr.split(":", 1)
                else:
                    var, default = expr, ""
                out += os.getenv(var, default)
                i = j + 1
            else:
                out += value[i]
                i += 1
        return out
    if isinstance(value, list):
        return [_env_interpolate(v) for v in value]
    if isinstance(value, dict):
        return {k: _env_interpolate(v) for k, v in value.items()}
    return value


@dataclass
class AppConfig:
    name: str
    version: str
    seed: int
    data_dir: str
    models_dir: str
    db_path: str
    alembic_ini: str


@dataclass
class ProviderConfig:
    default: str
    alpha_vantage: Dict[str, Any]
    dukascopy: Dict[str, Any]


@dataclass
class Instrument:
    symbol: str
    timeframes: List[str]


@dataclass
class FeaturesConfig:
    resample_rules: Dict[str, str]
    indicators: Dict[str, Any]
    standardization: Dict[str, Any]


@dataclass
class TrainingConfig:
    vae: Dict[str, Any]
    diffusion: Dict[str, Any]
    horizons_min: List[int]
    horizons_hours: List[int]
    horizons_days: List[int]
    batch_size: int
    lr: float
    max_epochs: int


@dataclass
class CalibrationConfig:
    alpha: float
    lambda_decay: float
    mondrian_by_session: bool
    credibility: Dict[str, float]


@dataclass
class ApiConfig:
    host: str
    port: int


@dataclass
class GuiConfig:
    polling_ms: int
    realtime_default: bool


@dataclass
class RootConfig:
    app: AppConfig
    providers: ProviderConfig
    instruments: List[Instrument]
    download: Dict[str, Any]
    features: FeaturesConfig
    training: TrainingConfig
    calibration: CalibrationConfig
    api: ApiConfig
    gui: GuiConfig


def load_config(path: str) -> RootConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = _env_interpolate(raw)
    app = AppConfig(**raw["app"])
    providers = ProviderConfig(**raw["providers"])
    instruments = [Instrument(**x) for x in raw.get("instruments", [])]
    features = FeaturesConfig(**raw["features"])
    training = TrainingConfig(**raw["training"])
    calibration = CalibrationConfig(**raw["calibration"])
    api = ApiConfig(**raw["api"])
    gui = GuiConfig(**raw["gui"])
    return RootConfig(
        app=app,
        providers=providers,
        instruments=instruments,
        download=raw["download"],
        features=features,
        training=training,
        calibration=calibration,
        api=api,
        gui=gui,
    )


def get_default_config_path() -> str:
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default_path = os.path.join(here, "configs", "default.yaml")
    return default_path


