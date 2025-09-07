from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


def compute_returns(df: pd.DataFrame) -> pd.Series:
    close = df["close_t"].astype(float)
    r = np.log(close).diff()
    return r


def rolling_std(r: pd.Series, n: int) -> pd.Series:
    return r.rolling(n, min_periods=n).std(ddof=1)


def ema(series: pd.Series, n: int) -> pd.Series:
    alpha = 2.0 / (n + 1)
    return series.ewm(alpha=alpha, adjust=False).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    high = df["high_t"].astype(float)
    low = df["low_t"].astype(float)
    close = df["close_t"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum.reduce([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ])
    return ema(pd.Series(tr, index=df.index), n)


def macd(close: pd.Series, nf: int, ns: int, n_sig: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_f = ema(close, nf)
    ema_s = ema(close, ns)
    m = ema_f - ema_s
    s = ema(m, n_sig)
    h = m - s
    return m, s, h


def rsi_wilder(close: pd.Series, n: int) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = ema(gain, n)
    avg_loss = ema(loss, n)
    rs = avg_gain / (avg_loss.replace({0.0: np.nan}))
    rsi = 100 - 100 / (1 + rs)
    return rsi


def bollinger(close: pd.Series, n: int, k: float) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=1)
    upper = ma + k * sd
    lower = ma - k * sd
    width = upper - lower
    p_bw = (close - lower) / width
    return upper, lower, width, p_bw


def keltner(close: pd.Series, atr_s: pd.Series, n: int, m: float) -> Tuple[pd.Series, pd.Series]:
    ema_c = ema(close, n)
    ku = ema_c + m * atr_s
    kl = ema_c - m * atr_s
    return ku, kl


def add_time_cyclicals(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["ts_utc"], unit="ms", utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    # Simple session dummies (UTC): Tokyo ~ [0,8), London ~ [8,16), NY ~ [13,21)
    df["session_tokyo"] = ((ts.dt.hour >= 0) & (ts.dt.hour < 8)).astype(int)
    df["session_london"] = ((ts.dt.hour >= 8) & (ts.dt.hour < 16)).astype(int)
    df["session_ny"] = ((ts.dt.hour >= 13) & (ts.dt.hour < 21)).astype(int)
    return df


def standardize_no_leak(train_df: pd.DataFrame, apply_df: pd.DataFrame, cols: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    stats: Dict[str, Tuple[float, float]] = {}
    out = apply_df.copy()
    for c in cols:
        mu = float(train_df[c].mean())
        sd = float(train_df[c].std(ddof=1))
        sd = sd if sd > 1e-12 else 1.0
        out[c] = (apply_df[c] - mu) / sd
        stats[c] = (mu, sd)
    return out, stats


def build_feature_frame(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    # Assumes validated input with required OHLCV columns and ts_utc
    df = df.sort_values("ts_utc").reset_index(drop=True)
    close = df["close_t"].astype(float)
    r = compute_returns(df)
    df["r_1"] = r
    df["sigma_20"] = rolling_std(r, 20)
    a = atr(df, cfg["indicators"].get("atr_n", 14))
    df["atr"] = a
    m, s, h = macd(close, cfg["indicators"].get("macd_nf", 12), cfg["indicators"].get("macd_ns", 26), cfg["indicators"].get("macd_sig", 9))
    df["macd"] = m
    df["macd_sig"] = s
    df["macd_hist"] = h
    df["rsi"] = rsi_wilder(close, cfg["indicators"].get("rsi_n", 14))
    bu, bl, bw, pb = bollinger(close, cfg["indicators"].get("bb_n", 20), float(cfg["indicators"].get("bb_k", 2.0)))
    df["bb_upper"] = bu
    df["bb_lower"] = bl
    df["bb_width"] = bw
    df["bb_percent_b"] = pb
    ku, kl = keltner(close, a, cfg["indicators"].get("keltner_n", 20), float(cfg["indicators"].get("keltner_m", 2.0)))
    df["kc_upper"] = ku
    df["kc_lower"] = kl
    df = add_time_cyclicals(df)
    return df


