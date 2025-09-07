from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def crps_samples(samples: np.ndarray, y: float) -> float:
    # samples: shape [N]
    s = np.asarray(samples, dtype=float).ravel()
    N = s.size
    if N == 0:
        return float("nan")
    term1 = np.mean(np.abs(s - y))
    # Efficient pairwise |xi-xj|
    s_sorted = np.sort(s)
    idx = np.arange(N)
    cumsum = np.cumsum(s_sorted)
    pair = (2 * idx - N + 1) * s_sorted
    term2 = (2.0 / (N * N)) * np.sum(pair)
    # Alternative classic: mean |xi-xj| = 2/(N^2) * sum_{k} (2k-N-1) s_(k)
    return term1 - 0.5 * term2


def empirical_quantiles(samples: np.ndarray, qs: Sequence[float]) -> List[float]:
    s = np.asarray(samples, dtype=float).ravel()
    return [float(np.quantile(s, q, method="linear")) for q in qs]


def pit_value(samples: np.ndarray, y: float) -> float:
    s = np.asarray(samples, dtype=float).ravel()
    return float(np.mean(s <= y))


def ks_pvalue_uniform(u: np.ndarray) -> float:
    # One-sample KS test for U(0,1), asymptotic p-value (no scipy)
    u = np.sort(np.asarray(u, dtype=float).ravel())
    n = u.size
    if n == 0:
        return float("nan")
    cdf = np.arange(1, n + 1) / n
    d_plus = np.max(cdf - u)
    d_minus = np.max(u - (np.arange(n) / n))
    d = max(d_plus, d_minus)
    # Asymptotic Kolmogorov distribution
    x = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    # p ≈ 2 ∑ (-1)^{k-1} e^{-2 k^2 x^2}
    p = 0.0
    for k in range(1, 101):
        p += (-1) ** (k - 1) * math.exp(-2 * (k * k) * (x * x))
    p *= 2.0
    return float(max(min(p, 1.0), 0.0))


