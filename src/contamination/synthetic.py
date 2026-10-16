"""Synthetic alpha-amylase fermentation batches with documented, injected anomalies.

SYNTHETIC DATA ONLY. The process model is a deliberately simple illustration
(logistic growth, oxygen demand proportional to growth rate, product formed in
stationary phase). It is not real alpha-amylase data or a validated contamination
model, and results on it show that the software behaves as intended, not real-world
detection performance.

Scenarios (each batch's ground truth is written to the separate outcomes table):

* ``normal``: ordinary batch-to-batch variation and measurement noise.
* ``slow_growth``: legitimate process variation (longer lag, slower growth), e.g. an
  older inoculum. Not contamination, but it may still look unusual.
* ``contamination``: a simulated bacterial contaminant (E. coli used as the example)
  grows from an onset time and adds biomass/turbidity, adds oxygen demand, acidifies
  the culture and lowers enzyme activity. The change is gradual. A visual/odour note
  (today's detection method) is recorded some hours later.
* ``sensor_spike``: a single faulty pH reading (measurement fault).
* ``do_probe_drift``: the dissolved-oxygen probe drifts downward (equipment fault);
  only the DO signal is affected, so its usual relationship with growth breaks.

Some batches also get a gap of missing online readings (``quality_flag="missing"``,
``value`` NaN); missing values are never filled with zero.

Genealogy: most contaminated batches share medium lot ``M-12``, a planted common
source for testing source tracing later. This is recorded in the outcomes table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import MEASUREMENT_COLUMNS, VARIABLES

DURATION_H = 48
# variable -> (sampling interval h, result availability delay h, source)
SAMPLING: dict[str, tuple[int, float, str]] = {
    "ph": (1, 0.0, "online_probe"),
    "temperature": (1, 0.0, "online_probe"),
    "dissolved_oxygen": (1, 0.0, "online_probe"),
    "biomass_od600": (2, 0.5, "manual_sample"),
    "enzyme_activity": (4, 3.0, "offline_assay"),
}
CONTAMINATED_LOT = "M-12"

SCENARIO_CATEGORY = {
    "normal": "normal",
    "slow_growth": "normal",
    "contamination": "contamination",
    "sensor_spike": "non_contamination_anomaly",
    "do_probe_drift": "non_contamination_anomaly",
}


@dataclass
class SyntheticDataset:
    measurements: pd.DataFrame
    outcomes: pd.DataFrame
    batches: pd.DataFrame

    def save(self, directory) -> None:
        from pathlib import Path

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.measurements.to_csv(path / "measurements.csv", index=False)
        self.outcomes.to_csv(path / "outcomes.csv", index=False)
        self.batches.to_csv(path / "batches.csv", index=False)


def _logistic(t: np.ndarray, x0: float, xmax: float, mu: float, lag: float) -> np.ndarray:
    tt = np.clip(t - lag, 0.0, None)
    return xmax / (1.0 + (xmax - x0) / x0 * np.exp(-mu * tt))


def _growth_rate(t: np.ndarray, x: np.ndarray, mu: float, xmax: float, lag: float) -> np.ndarray:
    return np.where(t >= lag, mu * x * (1.0 - x / xmax), 0.0)


def _simulate_batch(rng: np.random.Generator, scenario: str) -> tuple[dict, np.ndarray, dict]:
    """Return (true signals on the hourly grid, stage per hour, scenario info)."""
    t = np.arange(DURATION_H + 1, dtype=float)
    if scenario == "slow_growth":
        lag, mu = rng.uniform(7.5, 9.0), rng.normal(0.29, 0.015)
    else:
        lag, mu = rng.uniform(3.0, 5.0), rng.normal(0.36, 0.025)
    xmax, x0 = rng.normal(8.0, 0.35), 0.1
    amax = rng.normal(120.0, 6.0)

    x = _logistic(t, x0, xmax, mu, lag)
    r_norm = _growth_rate(t, x, mu, xmax, lag) / (mu * xmax / 4.0)  # 0..1
    ph = 7.0 - 0.25 * (x / xmax) - 0.10 * r_norm
    temp = np.full_like(t, 37.0 + rng.normal(0.0, 0.05))
    do = 100.0 - 70.0 * r_norm - 12.0 * (x / xmax)
    stationary_start = lag + np.log(9.0 * (xmax - x0) / x0) / mu  # x reaches 0.9 xmax
    enzyme = amax / (1.0 + np.exp(-(t - (stationary_start + 2.0)) / 3.0))
    od = x.copy()

    info: dict = {"onset_h": np.nan, "reference_event_h": np.nan, "observation_h": None}
    if scenario == "contamination":
        t0 = rng.uniform(14.0, 24.0)
        cmax, mu_c = rng.uniform(2.5, 3.5), rng.uniform(0.45, 0.6)
        c = np.where(t >= t0, _logistic(t, 0.02, cmax, mu_c, t0), 0.0)
        rc_norm = _growth_rate(t, c, mu_c, cmax, t0) / (mu_c * cmax / 4.0)
        od = od + c
        do = do - 25.0 * rc_norm - 6.0 * (c / cmax)
        ph = ph - 0.15 * (c / cmax)
        enzyme = enzyme * (1.0 - 0.4 * (c / cmax))
        obs_h = float(np.round(t0 + rng.uniform(12.0, 18.0)))
        info["onset_h"] = round(float(t0), 2)
        if obs_h <= DURATION_H:
            info["observation_h"] = obs_h
            info["reference_event_h"] = obs_h
        else:  # not noticed during the run; found at end-of-batch QC
            info["reference_event_h"] = float(DURATION_H)
    elif scenario == "sensor_spike":
        ts = int(rng.integers(10, 41))
        info["onset_h"] = float(ts)
        info["spike"] = (ts, rng.uniform(0.5, 0.8))
    elif scenario == "do_probe_drift":
        t0 = float(rng.integers(16, 31))
        info["onset_h"] = t0
        info["do_drift_from"] = t0

    stage = np.where(t < lag, "lag", np.where(x < 0.9 * xmax, "exponential", "stationary"))
    signals = {
        "ph": ph,
        "temperature": temp,
        "dissolved_oxygen": np.clip(do, 2.0, 100.0),
        "biomass_od600": od,
        "enzyme_activity": enzyme,
    }
    return signals, stage, info


def _noisy(rng: np.random.Generator, variable: str, true: np.ndarray) -> np.ndarray:
    if variable == "ph":
        return true + rng.normal(0, 0.02, true.shape)
    if variable == "temperature":
        return true + rng.normal(0, 0.06, true.shape)
    if variable == "dissolved_oxygen":
        return np.clip(true + rng.normal(0, 1.5, true.shape), 0.0, 100.0)
    if variable == "biomass_od600":
        return np.clip(
            true * (1 + rng.normal(0, 0.04, true.shape)) + rng.normal(0, 0.02, true.shape), 0, None
        )
    if variable == "enzyme_activity":
        return np.clip(
            true * (1 + rng.normal(0, 0.05, true.shape)) + rng.normal(0, 1.0, true.shape), 0, None
        )
    raise KeyError(variable)


DECIMALS = {
    "ph": 3,
    "temperature": 2,
    "dissolved_oxygen": 1,
    "biomass_od600": 3,
    "enzyme_activity": 1,
}


def _batch_rows(rng, batch_id, start, scenario, missing_gap: bool) -> tuple[list[dict], dict]:
    signals, stage, info = _simulate_batch(rng, scenario)
    gap = None
    if missing_gap:
        var = str(rng.choice(["ph", "dissolved_oxygen"]))
        g0 = int(rng.integers(8, 36))
        gap = (var, g0, g0 + int(rng.integers(4, 9)))
    rows = []
    for variable, (interval, delay, source) in SAMPLING.items():
        hours = np.arange(0, DURATION_H + 1, interval)
        values = _noisy(rng, variable, signals[variable][hours])
        if variable == "ph" and "spike" in info:
            ts, size = info["spike"]
            values = np.where(hours == ts, values + size, values)
        if variable == "dissolved_oxygen" and "do_drift_from" in info:
            drift = np.clip(hours - info["do_drift_from"], 0, None) * 1.2
            values = np.clip(values - drift, 0.0, 100.0)
        for h, v in zip(hours, values):
            flag, value = "valid", round(float(v), DECIMALS[variable])
            if gap and variable == gap[0] and gap[1] <= h < gap[2]:
                flag, value = "missing", np.nan
            observation = None
            if (
                variable == "biomass_od600"
                and info["observation_h"] is not None
                and h == info["observation_h"]
            ):
                observation = "Off-colour broth and unusual odour noted (simulated)"
            ts = start + pd.Timedelta(hours=int(h))
            rows.append(
                {
                    "batch_id": batch_id,
                    "timestamp": ts,
                    "available_at": ts + pd.Timedelta(hours=delay),
                    "stage": stage[h],
                    "variable": variable,
                    "value": value,
                    "unit": VARIABLES[variable].unit,
                    "source": source,
                    "quality_flag": flag,
                    "observation": observation,
                }
            )
    return rows, info


def generate_dataset(
    n_reference: int = 30,
    n_test_normal: int = 10,
    n_slow_growth: int = 3,
    n_contamination: int = 6,
    n_sensor_spike: int = 3,
    n_do_drift: int = 3,
    missing_gap_fraction: float = 0.2,
    seed: int = 7,
    start: str = "2026-01-05 08:00",
) -> SyntheticDataset:
    """Generate a reproducible synthetic dataset.

    The first ``n_reference`` batches (chronologically) are all normal and are intended
    as the reference set; the later batches are a shuffled mix of all scenarios for
    testing, which gives a chronological, whole-batch split.
    """
    rng = np.random.default_rng(seed)
    test = (
        ["normal"] * n_test_normal
        + ["slow_growth"] * n_slow_growth
        + ["contamination"] * n_contamination
        + ["sensor_spike"] * n_sensor_spike
        + ["do_probe_drift"] * n_do_drift
    )
    rng.shuffle(test)
    scenarios = ["normal"] * n_reference + list(test)

    t_start = pd.Timestamp(start)
    contaminated_seen = 0
    all_rows, outcomes, batches = [], [], []
    for i, scenario in enumerate(scenarios):
        batch_id = f"B{i + 1:03d}"
        b_start = t_start + pd.Timedelta(days=2 * i)
        rows, info = _batch_rows(
            rng, batch_id, b_start, scenario, rng.random() < missing_gap_fraction
        )
        all_rows.extend(rows)

        medium_lot = f"M-{i // 3 + 1:02d}"
        if medium_lot == CONTAMINATED_LOT:
            medium_lot = "M-12a"  # keep the planted lot exclusive to contaminated batches
        simulated_source = ""
        if scenario == "contamination":
            if contaminated_seen < max(1, round(n_contamination * 2 / 3)):
                medium_lot, simulated_source = CONTAMINATED_LOT, f"medium lot {CONTAMINATED_LOT}"
            else:
                simulated_source = "independent (not lot-related)"
            contaminated_seen += 1
        batches.append(
            {
                "batch_id": batch_id,
                "start_time": b_start,
                "bioreactor_id": f"BR-{i % 2 + 1}",
                "medium_lot": medium_lot,
                "inoculum_lot": f"I-{i // 5 + 1:02d}",
            }
        )
        reasons = {
            "normal": "Simulated normal batch",
            "slow_growth": "Simulated legitimate slow growth (longer lag, lower growth rate)",
            "contamination": f"Simulated bacterial contaminant from {info['onset_h']} h",
            "sensor_spike": f"Simulated single faulty pH reading at {info['onset_h']} h",
            "do_probe_drift": f"Simulated DO probe drift from {info['onset_h']} h",
        }
        outcomes.append(
            {
                "batch_id": batch_id,
                "category": SCENARIO_CATEGORY[scenario],
                "scenario": scenario,
                "outcome_status": "confirmed",
                "onset_h": info["onset_h"],
                "reference_event_h": info["reference_event_h"],
                "simulated_source": simulated_source,
                "reason": reasons[scenario],
                "data_origin": "synthetic",
            }
        )

    measurements = pd.DataFrame(all_rows, columns=MEASUREMENT_COLUMNS)
    return SyntheticDataset(measurements, pd.DataFrame(outcomes), pd.DataFrame(batches))
