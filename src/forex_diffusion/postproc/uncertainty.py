from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


def weighted_quantile(values: np.ndarray, quantile: float, sample_weight: np.ndarray) -> float:
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(sample_weight, dtype=float).ravel()
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    if cw[-1] == 0:
        return float(np.median(v))
    target = quantile * cw[-1]
    idx = np.searchsorted(cw, target, side="left")
    idx = min(max(idx, 0), v.size - 1)
    return float(v[idx])


@dataclass
class ConformalCalibrator:
    alpha: float
    lambda_decay: float
    mondrian_by_session: bool = True

    def compute_nonconformity(self, q05: np.ndarray, q50: np.ndarray, q95: np.ndarray, y: np.ndarray) -> np.ndarray:
        # s_i = interval excess wrt observed y
        return np.maximum(q05 - y, 0) + np.maximum(y - q95, 0)

    def weights(self, ts_now: np.ndarray, ts_hist: np.ndarray) -> np.ndarray:
        dt = np.maximum(0.0, ts_now - ts_hist)
        w = np.exp(-self.lambda_decay * dt)
        if w.sum() == 0:
            return np.ones_like(w)
        return w / w.sum()

    def adjust_intervals(self, q05: float, q50: float, q95: float, s_hist: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
        delta = weighted_quantile(s_hist, 1 - self.alpha, w)
        delta_base = (q95 - q05) / 2.0
        low = q05 - delta
        high = q95 + delta
        return float(low), float(high)


def credibility_score(cov_obs: float, target: float, width_norm: float, entropy: float, d_regime: float,
                      a: float, b: float, c: float, d: float) -> float:
    import math
    z = a * abs(cov_obs - target) + b * max(0.0, width_norm - 1.0) + c * entropy + d * abs(d_regime)
    return 1.0 / (1.0 + math.exp(z))


