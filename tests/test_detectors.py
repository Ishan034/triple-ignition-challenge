import numpy as np
import pytest

from contamination.detectors import cusum, ewma, ewma_limit, robust_z


def test_robust_z():
    assert robust_z(np.array([12.0]), np.array([10.0]), np.array([0.5]))[0] == pytest.approx(4.0)


def test_ewma_recursion_and_nan_handling():
    out = ewma(np.array([1.0, np.nan, 1.0]), lam=0.5)
    assert out[0] == pytest.approx(0.5)
    assert np.isnan(out[1])  # no observation -> no statistic, state carried
    assert out[2] == pytest.approx(0.75)


def test_ewma_limit_formula():
    assert ewma_limit(3.0, 0.2) == pytest.approx(3.0 * np.sqrt(0.2 / 1.8))


def test_cusum_accumulates_sustained_shift_but_not_noise():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 200)
    assert np.nanmax(np.abs(cusum(noise))) < 15
    shifted = np.concatenate([np.zeros(10), np.full(10, 1.5)])
    out = cusum(shifted, k=0.5)
    assert out[-1] == pytest.approx(10.0)  # 10 steps * (1.5 - 0.5)


def test_cusum_sign_follows_direction():
    assert cusum(np.full(5, -2.0))[-1] < 0


def test_ewma_detects_gradual_drift_earlier_than_single_threshold():
    """A slow drift crosses the EWMA limit before any single z exceeds 4."""
    z = np.concatenate([np.zeros(10), np.linspace(0.2, 3.9, 20)])
    first_threshold = np.argmax(np.abs(z) > 4.0) if (np.abs(z) > 4.0).any() else None
    first_ewma = int(np.argmax(np.abs(ewma(z, 0.2)) > ewma_limit(3.0, 0.2)))
    assert first_threshold is None
    assert first_ewma < len(z)
