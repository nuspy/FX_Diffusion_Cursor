from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    sharpe: float
    max_dd: float
    turnover: float


def compute_metrics(returns: pd.Series, bars_per_year: int) -> Tuple[float, float]:
    mu = returns.mean()
    sd = returns.std(ddof=1)
    sharpe = (mu / (sd + 1e-12)) * np.sqrt(bars_per_year)
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = (peak - equity) / peak
    mdd = float(dd.max()) if len(dd) else 0.0
    return float(sharpe), mdd


def strategy_from_quantiles(df: pd.DataFrame, threshold: float = 0.0, max_hold: int = 20) -> pd.Series:
    # Simple: long if q50 > close, short if q50 < close; exit after max_hold or flip
    pos = np.zeros(len(df), dtype=float)
    hold = 0
    for i in range(1, len(df)):
        if hold > 0:
            hold -= 1
        signal = 0.0
        if df["q50"].iloc[i] > df["close_t"].iloc[i] * (1 + threshold):
            signal = 1.0
        elif df["q50"].iloc[i] < df["close_t"].iloc[i] * (1 - threshold):
            signal = -1.0
        if signal != 0:
            pos[i] = signal
            hold = max_hold
        elif hold > 0:
            pos[i] = pos[i-1]
        else:
            pos[i] = 0.0
    return pd.Series(pos, index=df.index)


def evaluate(df: pd.DataFrame, spread: float = 0.0, slippage: float = 0.0, bars_per_year: int = 252*24*60) -> BacktestResult:
    pos = strategy_from_quantiles(df)
    px = df["close_t"].astype(float)
    r = px.pct_change().fillna(0.0)
    trade_cost = (np.abs(np.diff(pos, prepend=0)) > 0).astype(float) * (spread + slippage)
    ret = pos * r - trade_cost
    sharpe, mdd = compute_metrics(ret, bars_per_year)
    to = np.mean(np.abs(np.diff(pos, prepend=0)))
    eq = (1 + ret).cumprod()
    return BacktestResult(equity=eq, sharpe=sharpe, max_dd=mdd, turnover=float(to))


