"""Uncertainty ranges for evaluation results.

Evaluation counts are small (10 fault batches, 30 normal batches), so a bare "8/10"
invites over-reading. Every count is reported with a 95% Wilson score interval, and
medians of timings with a bootstrap interval over batches.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def wilson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (well behaved at 0/n and n/n)."""
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(0.5 + confidence / 2)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> str:
    """'8/10 (49-94%)': the count with its 95% interval."""
    if n == 0:
        return "0/0"
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * lo:.0f}–{100 * hi:.0f}%)"


def median_interval(
    values, confidence: float = 0.95, n_boot: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """Median and a percentile-bootstrap interval, resampling batches."""
    x = np.asarray([v for v in values if np.isfinite(v)], float)
    if not len(x):
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    a = (1 - confidence) / 2
    return (float(np.median(x)), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a)))


def median_text(values, unit: str = "h") -> str:
    m, lo, hi = median_interval(values)
    if not np.isfinite(m):
        return "n/a"
    return f"{m:.3g} {unit} (95% range {lo:.3g}–{hi:.3g} {unit})"
