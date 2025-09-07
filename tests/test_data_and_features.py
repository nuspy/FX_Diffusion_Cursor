import os
import pandas as pd

from forex_diffusion.features.pipeline import build_feature_frame
from forex_diffusion.services.marketdata import validate_candles


def make_dummy_df(n=100):
    ts = pd.Series(range(1_600_000_000_000, 1_600_000_000_000 + n * 60_000, 60_000))
    df = pd.DataFrame({
        'ts_utc': ts,
        'open_t': 1.10,
        'high_t': 1.11,
        'low_t': 1.09,
        'close_t': 1.10 + (pd.Series(range(n)) * 1e-5),
        'volume_t': 100.0,
        'symbol': 'EURUSD',
        'timeframe': '1m',
        'resampled': False,
    })
    return validate_candles(df)


def test_build_feature_frame_basic():
    df = make_dummy_df(120)
    cfg = {
        'indicators': {
            'atr_n': 14,
            'macd_nf': 12,
            'macd_ns': 26,
            'macd_sig': 9,
            'rsi_n': 14,
            'bb_n': 20,
            'bb_k': 2.0,
            'keltner_n': 20,
            'keltner_m': 2.0,
        }
    }
    out = build_feature_frame(df, cfg)
    for col in [
        'r_1','sigma_20','atr','macd','macd_sig','macd_hist','rsi','bb_upper','bb_lower','bb_width','bb_percent_b','kc_upper','kc_lower','hour_sin','hour_cos','session_tokyo','session_london','session_ny'
    ]:
        assert col in out.columns


