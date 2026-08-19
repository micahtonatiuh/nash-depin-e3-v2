"""
fairness_metrics.py — Gini and Jain (identical to E2 for direct comparability).
"""
import numpy as np
from typing import Sequence


def gini_coefficient(rewards: Sequence[float]) -> float:
    """Gini of the reward distribution. 0=fair, ->1=monopoly.
    Shifts by the minimum if negatives are present (classical Gini assumes
    non-negative values)."""
    x = np.asarray(rewards, dtype=float)
    n = len(x)
    if n == 0:
        return 0.0
    if x.min() < 0:
        x = x - x.min()
    if x.sum() <= 1e-12:
        return 0.0
    diff = np.abs(x[:, None] - x[None, :]).sum()
    return float(np.clip(diff / (2.0 * n * n * x.mean()), 0.0, 1.0))


def jain_fairness_index(rewards: Sequence[float]) -> float:
    """Jain (1984): (sum x)^2 / (n * sum x^2). 1=fair, 1/n=monopoly."""
    x = np.asarray(rewards, dtype=float)
    n = len(x)
    if n == 0:
        return 1.0
    if x.min() < 0:
        x = x - x.min()
    ss = np.sum(x ** 2)
    if ss <= 1e-12:
        return 1.0
    return float(np.clip((x.sum() ** 2) / (n * ss), 1.0 / n, 1.0))
