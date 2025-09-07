from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..services.marketdata import MarketDataService, validate_candles


log = logging.getLogger(__name__)


def read_candles_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ts_utc" in df.columns:
        df = df.sort_values("ts_utc")
    return validate_candles(df)


def read_candles_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts_utc" in df.columns:
        df = df.sort_values("ts_utc")
    return validate_candles(df)


def timeframe_ms(tf: str) -> int:
    mapping = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "60m": 3_600_000, "1d": 86_400_000}
    if tf not in mapping:
        raise ValueError(f"unsupported timeframe {tf}")
    return mapping[tf]


def timeframe_rule(tf: str) -> str:
    mapping = {"1m": "1T", "5m": "5T", "15m": "15T", "60m": "60T", "1d": "1D"}
    if tf not in mapping:
        raise ValueError(f"unsupported timeframe {tf}")
    return mapping[tf]


def causal_resample(df_1m: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    if target_tf == "1m":
        return df_1m.copy()
    if "ts_utc" not in df_1m.columns:
        raise ValueError("ts_utc required in df_1m")
    # Use datetime index for resample
    ts = pd.to_datetime(df_1m["ts_utc"], unit="ms", utc=True)
    df = df_1m.set_index(ts)
    agg = {
        "open_t": "first",
        "high_t": "max",
        "low_t": "min",
        "close_t": "last",
        "volume_t": "sum",
        "symbol": "first",
        "timeframe": "first",
        "resampled": "first",
    }
    out = df.resample(timeframe_rule(target_tf), label="right", closed="right").agg(agg)
    out["timeframe"] = target_tf
    out["resampled"] = True
    out = out.dropna(subset=["open_t", "high_t", "low_t", "close_t"])  # ensure formed bars
    out["ts_utc"] = (out.index.view("int64") // 1_000_000).astype("int64")
    return out.reset_index(drop=True)


def get_last_ts(engine: Engine, symbol: str, timeframe: str) -> Optional[int]:
    q = text(
        "SELECT MAX(ts_utc) as last_ts FROM market_data WHERE symbol = :s AND timeframe = :tf"
    )
    with engine.connect() as conn:
        row = conn.execute(q, {"s": symbol, "tf": timeframe}).mappings().first()
        return int(row["last_ts"]) if row and row["last_ts"] is not None else None


def upsert_candles(engine: Engine, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cols = [
        "symbol",
        "timeframe",
        "ts_utc",
        "open_t",
        "high_t",
        "low_t",
        "close_t",
        "volume_t",
        "resampled",
    ]
    insert_sql = text(
        """
        INSERT INTO market_data (symbol, timeframe, ts_utc, open_t, high_t, low_t, close_t, volume_t, resampled)
        VALUES (:symbol, :timeframe, :ts_utc, :open_t, :high_t, :low_t, :close_t, :volume_t, :resampled)
        ON CONFLICT(symbol, timeframe, ts_utc) DO UPDATE SET
            open_t=excluded.open_t,
            high_t=excluded.high_t,
            low_t=excluded.low_t,
            close_t=excluded.close_t,
            volume_t=excluded.volume_t,
            resampled=excluded.resampled
        """
    )
    recs = df[cols].to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(insert_sql, recs)
    return len(recs)


def _qa_log_gaps(df: pd.DataFrame, timeframe: str) -> Tuple[int, int]:
    if df.empty:
        return 0, 0
    ms = timeframe_ms(timeframe)
    s = df.sort_values("ts_utc")["ts_utc"].to_numpy()
    gaps = ((s[1:] - s[:-1]) > ms).sum() if len(s) > 1 else 0
    outliers = ((df[["high_t", "low_t", "open_t", "close_t"]].isna()).any(axis=1)).sum()
    if gaps or outliers:
        log.warning("QA: timeframe=%s gaps=%s outliers_na=%s", timeframe, gaps, outliers)
    else:
        log.info("QA: timeframe=%s OK (no gaps/outliers)", timeframe)
    return int(gaps), int(outliers)


def fetch_alpha_vantage(symbol: str, mds: MarketDataService) -> Dict[str, pd.DataFrame]:
    daily = mds.fetch_fx_daily(symbol)
    intraday = mds.fetch_fx_intraday(symbol, interval="1min", outputsize="full")
    return {"1d": daily, "1m": intraday}


def backfill_startup(engine: Engine, mds: MarketDataService, instruments: List[Dict[str, List[str]]]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for inst in instruments:
        symbol = inst["symbol"]
        tfs = inst["timeframes"]
        fetched = fetch_alpha_vantage(symbol, mds)
        # Insert 1d
        if "1d" in tfs:
            last_ts = get_last_ts(engine, symbol, "1d")
            df = fetched["1d"]
            if last_ts is not None:
                df = df[df["ts_utc"] > last_ts]
            df = validate_candles(df)
            ins = upsert_candles(engine, df)
            _qa_log_gaps(df, "1d")
            summary.setdefault(symbol, {})["1d"] = ins
        # Insert 1m
        if "1m" in tfs:
            last_ts = get_last_ts(engine, symbol, "1m")
            df = fetched["1m"]
            if last_ts is not None:
                df = df[df["ts_utc"] > last_ts]
            df = validate_candles(df)
            ins = upsert_candles(engine, df)
            _qa_log_gaps(df, "1m")
            summary.setdefault(symbol, {})["1m"] = ins
        # Resample from 1m to higher TFs
        if "1m" in fetched:
            base_1m = fetched["1m"]
            for tf in ["5m", "15m", "60m"]:
                if tf in tfs:
                    last_ts = get_last_ts(engine, symbol, tf)
                    rs = causal_resample(base_1m, tf)
                    if last_ts is not None:
                        rs = rs[rs["ts_utc"] > last_ts]
                    ins = upsert_candles(engine, rs)
                    _qa_log_gaps(rs, tf)
                    summary.setdefault(symbol, {})[tf] = ins
    return summary


def split_walk_forward(df: pd.DataFrame, train_end: int, val_end: int, test_end: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # All bounds are ts_utc (ms), half-open intervals [)
    train = df[(df["ts_utc"] < train_end)]
    val = df[(df["ts_utc"] >= train_end) & (df["ts_utc"] < val_end)]
    test = df[(df["ts_utc"] >= val_end) & (df["ts_utc"] < test_end)]
    return train, val, test


