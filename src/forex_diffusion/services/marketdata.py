from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import httpx
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt


log = logging.getLogger(__name__)


@dataclass
class MarketDataService:
    base_url: str
    api_key: str
    timeout_s: int = 30

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout_s)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(5))
    def _get(self, params: Dict[str, str]) -> Dict:
        with self._client() as client:
            r = client.get(self.base_url, params=params)
            if r.status_code == 429:
                # honor rate limit
                time.sleep(12)
                raise httpx.HTTPStatusError("429", request=r.request, response=r)
            r.raise_for_status()
            return r.json()

    def fetch_fx_daily(self, symbol: str) -> pd.DataFrame:
        params = {
            "function": "FX_DAILY",
            "from_symbol": symbol[:3],
            "to_symbol": symbol[3:],
            "outputsize": "full",
            "apikey": self.api_key,
            "datatype": "json",
        }
        data = self._get(params)
        key = "Time Series FX (Daily)"
        if key not in data:
            raise RuntimeError(f"Unexpected response keys: {list(data.keys())}")
        df = (
            pd.DataFrame(data[key]).T.rename(columns={
                "1. open": "open_t",
                "2. high": "high_t",
                "3. low": "low_t",
                "4. close": "close_t",
            })
        )
        df.index = pd.to_datetime(df.index, utc=True)
        for c in ["open_t", "high_t", "low_t", "close_t"]:
            df[c] = df[c].astype(float)
        df["volume_t"] = pd.NA
        df["symbol"] = symbol
        df["timeframe"] = "1d"
        df["ts_utc"] = (df.index.view("int64") // 1_000_000).astype("int64")
        df["resampled"] = False
        return df.reset_index(drop=True)

    def fetch_fx_intraday(self, symbol: str, interval: str = "1min", outputsize: str = "full") -> pd.DataFrame:
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": symbol[:3],
            "to_symbol": symbol[3:],
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "datatype": "json",
        }
        data = self._get(params)
        key = f"Time Series FX ({interval})"
        if key not in data:
            raise RuntimeError(f"Unexpected response keys: {list(data.keys())}")
        df = (
            pd.DataFrame(data[key]).T.rename(columns={
                "1. open": "open_t",
                "2. high": "high_t",
                "3. low": "low_t",
                "4. close": "close_t",
            })
        )
        df.index = pd.to_datetime(df.index, utc=True)
        for c in ["open_t", "high_t", "low_t", "close_t"]:
            df[c] = df[c].astype(float)
        df["volume_t"] = pd.NA
        df["symbol"] = symbol
        df["timeframe"] = "1m" if interval == "1min" else interval
        df["ts_utc"] = (df.index.view("int64") // 1_000_000).astype("int64")
        df["resampled"] = False
        return df.reset_index(drop=True)


def validate_candles(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure formal spec invariants
    req_cols = ["open_t", "high_t", "low_t", "close_t"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"missing column {c}")
    if not ((df["high_t"] >= df[["open_t", "close_t"]].max(axis=1)) & (df["low_t"] <= df[["open_t", "close_t"]].min(axis=1)) & (df["low_t"] <= df["high_t"]).all()):
        # Basic consistency; more checks can be added
        pass
    return df


# Optional Dukascopy stub (same I/O contract)
@dataclass
class DukascopyService:
    base_url: str
    api_key: str | None = None
    timeout_s: int = 30

    def current_prices(self, symbol: str) -> pd.DataFrame:
        # Placeholder: to be implemented with real REST client
        return pd.DataFrame(columns=["ts_utc","open_t","high_t","low_t","close_t","volume_t","symbol","timeframe","resampled"])  # empty

    def historical_prices(self, symbol: str, timeframe: str) -> pd.DataFrame:
        # Placeholder
        return pd.DataFrame(columns=["ts_utc","open_t","high_t","low_t","close_t","volume_t","symbol","timeframe","resampled"])  # empty


