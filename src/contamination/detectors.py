"""Layer 4, anomaly detection statistics.

Each detector turns a sequence of robust z-scores for one variable in one batch
(ordered by measurement time, valid observations only) into a signed monitoring
statistic. An alert is raised where ``|statistic| > limit``.

* ``robust_threshold``: the robust z itself, ``z = (x - median) / (1.4826 * MAD)``.
  Reacts to large single deviations.
* ``ewma``: ``E_t = lambda * z_t + (1 - lambda) * E_{t-1}``, with ``E_0 = 0``. Reacts to
  smaller sustained shifts. Default limit ``L * sqrt(lambda / (2 - lambda))``.
* ``cusum``: two-sided tabular CUSUM, ``S+_t = max(0, S+_{t-1} + z_t - k)`` and
  ``S-_t = max(0, S-_{t-1} - z_t - k)``. Accumulates evidence of a persistent shift.
* ``shift``: a CUSUM of each reading's change from the batch's *own* recent past,
  ``d_t = z_t - median(z over the 24 h ending 12 h before t)``. Every batch sits a
  little above or below the reference for its whole run (inoculum, recipe, control
  strategy); that offset cancels in ``d``, so a change that starts partway through a
  batch, such as a contaminant taking hold, stands out. The 12 h lag keeps a slowly
  growing change out of its own baseline.
* ``linked_shift``: the same change-from-own-past, smoothed over 6 readings and scaled
  by its normal spread, for a group of physically linked signals (e.g. CO2 output,
  oxygen uptake, dissolved oxygen). The statistic is the *second-largest* change in the
  group, so it only rises when at least two linked signals change together, which
  ordinary mid-batch wobble in one signal does not do. Computed in ``monitor.score``.
* ``loop_check``: for a controlled variable (pH, temperature) whose controller output
  is recorded, the robust z of the variable at readings where the controller is *not*
  pushing back (its output's z in the correcting direction is below a small margin),
  and 0 where it is. A healthy loop answers a drift by working harder; a drift with no
  response points at the actuator. Computed in ``monitor.score`` because it needs two
  variables.

These are measures of how unusual the process is. None of them is a probability of
contamination.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALGORITHM_VERSION = "0.4.0"


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    description: str


DETECTORS: dict[str, DetectorSpec] = {
    "robust_threshold": DetectorSpec(
        "robust_threshold", "Single reading far from the reference median (robust z-score)"
    ),
    "ewma": DetectorSpec("ewma", "Sustained shift in the smoothed (EWMA) deviation"),
    "cusum": DetectorSpec("cusum", "Accumulated persistent deviation (two-sided CUSUM)"),
    "shift": DetectorSpec("shift", "Change from the batch's own recent past (CUSUM)"),
    "linked_shift": DetectorSpec(
        "linked_shift", "At least two linked signals changed together from the batch's own past"
    ),
    "loop_check": DetectorSpec(
        "loop_check", "Controlled variable off its reference while its controller does not respond"
    ),
}


def robust_z(value: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(value, float) - np.asarray(median, float)) / np.asarray(scale, float)


def ewma(z: np.ndarray, lam: float = 0.2) -> np.ndarray:
    """EWMA of z. NaN inputs (no reference) leave the state unchanged and output NaN."""
    out = np.full(len(z), np.nan)
    e = 0.0
    for i, zi in enumerate(z):
        if np.isfinite(zi):
            e = lam * zi + (1.0 - lam) * e
            out[i] = e
    return out


def ewma_limit(L: float = 3.0, lam: float = 0.2) -> float:
    """Asymptotic EWMA control limit for standardised (unit-variance) input."""
    return L * np.sqrt(lam / (2.0 - lam))


def cusum(z: np.ndarray, k: float = 0.5) -> np.ndarray:
    """Signed two-sided CUSUM: +S+ when the upward sum dominates, -S- otherwise."""
    out = np.full(len(z), np.nan)
    sp = sm = 0.0
    for i, zi in enumerate(z):
        if np.isfinite(zi):
            sp = max(0.0, sp + zi - k)
            sm = max(0.0, sm - zi - k)
            out[i] = sp if sp >= sm else -sm
    return out


def loop_check(z_controlled: np.ndarray, push: np.ndarray, response_z: float) -> np.ndarray:
    """Controlled-variable z where the controller is not pushing back, else 0.

    ``push`` is the controller output's z in the direction that would correct the
    current deviation. NaN where either is unknown.
    """
    zc, pu = np.asarray(z_controlled, float), np.asarray(push, float)
    out = np.where(pu < response_z, zc, 0.0)
    out[~(np.isfinite(zc) & np.isfinite(pu))] = np.nan
    return out


def own_baseline_change(
    z: np.ndarray, t: np.ndarray, lag_h: float = 12.0, window_h: float = 24.0, min_points: int = 3
) -> np.ndarray:
    """z minus the median z of the same batch over (t - lag - window, t - lag]."""
    z, t = np.asarray(z, float), np.asarray(t, float)
    out = np.full(len(z), np.nan)
    ok = np.isfinite(z)
    for i in np.flatnonzero(ok):
        sel = ok & (t <= t[i] - lag_h) & (t > t[i] - lag_h - window_h)
        if sel.sum() >= min_points:
            out[i] = z[i] - np.median(z[sel])
    return out


def statistics(
    z: np.ndarray, ewma_lambda: float, cusum_k: float, t: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    out = {
        "robust_threshold": np.asarray(z, float),
        "ewma": ewma(z, ewma_lambda),
        "cusum": cusum(z, cusum_k),
    }
    if t is not None:
        out["shift"] = cusum(own_baseline_change(z, t), cusum_k)
    return out


def default_limits(ewma_lambda: float, ewma_L: float, robust_limit: float, cusum_h: float) -> dict:
    return {
        "robust_threshold": robust_limit,
        "ewma": ewma_limit(ewma_L, ewma_lambda),
        "cusum": cusum_h,
        "shift": cusum_h,
        "linked_shift": robust_limit,
        "loop_check": robust_limit,
    }
