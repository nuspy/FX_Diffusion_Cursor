from __future__ import annotations

import argparse
import logging
from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from ..utils.config import load_config, get_default_config_path
from ..utils.logging import setup_logging


log = logging.getLogger(__name__)
app = FastAPI(title="MagicForex Inference API")


class ForecastRequest(BaseModel):
    symbol: str
    timeframe: str
    horizons: List[int]
    N_samples: int = 64
    apply_conformal: bool = True


@app.post("/forecast")
def forecast(req: ForecastRequest) -> Dict:
    # Placeholder: return a stub response; to be wired with real model later
    quantiles = {str(h): {"q05": 0.0, "q50": 0.0, "q95": 0.0} for h in req.horizons}
    bands = {str(h): {"low": -0.1, "high": 0.1} for h in req.horizons}
    diag = {"CRPS_model": None, "CRPS_RW": None, "PIT_pvalue": None}
    cred = {str(h): 0.5 for h in req.horizons}
    return {
        "quantiles": quantiles,
        "bands_conformal": bands,
        "credibility": cred,
        "diagnostics": diag,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=get_default_config_path())
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(log_dir=cfg.app.data_dir, app_name=cfg.app.name)
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)


if __name__ == "__main__":
    main()


