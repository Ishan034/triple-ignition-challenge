"""Phase I (fit) and Phase II (monitor) for the anomaly-detection layer.

* :func:`fit_reference` builds the reference from approved normal batches and
  calibrates each detector's limit per variable. The calibration is leave-one-out:
  each approved batch is scored against a reference built *without* it, giving one
  peak statistic per batch. The limit is a margin above a high percentile of those
  peaks (``calibration_quantile``; 1.0 = the single worst batch), and never below the
  textbook default. The limits are then frozen into the versioned profile.
* :func:`monitor` scores batches against a frozen profile and returns structured
  alerts. With ``as_of_h`` it only uses results available by that elapsed hour, as
  the system would have seen them at that point in the batch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .alerts import Alert
from .detectors import ALGORITHM_VERSION, DETECTORS, default_limits, statistics
from .ingest import valid_values
from .reference import ReferenceProfile, build_reference
from .schema import VARIABLES

NO_EVIDENCE = "none recorded (no independent test results supplied)"


@dataclass(frozen=True)
class MonitorConfig:
    window_h: float = 1.0
    min_reference_batches: int = 5
    robust_limit: float = 4.0
    ewma_lambda: float = 0.2
    ewma_L: float = 3.0
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    calibration_margin: float = 1.1
    # Percentile of the approved batches' leave-one-out peaks that sets each limit.
    # 1.0 takes the single worst approved batch, so one outlier in the reference
    # widens the limit for everyone; a lower value accepts that a few approved
    # batches would have alerted, in exchange for tighter limits. The robust threshold
    # reacts to single readings, so its worst batch is usually one glitch; EWMA and
    # CUSUM accumulate, so their peaks reflect a whole trajectory. Hence two settings.
    calibration_quantile: float = 1.0  # EWMA and CUSUM
    threshold_calibration_quantile: float = 1.0  # robust threshold (starting point)
    # Share of approved batches allowed to alert in leave-one-out calibration. When set,
    # the robust-threshold limit is tightened from the quantile above until this budget
    # is reached. None keeps the quantiles as given.
    false_alarm_budget: float | None = 0.05
    corroboration_window_h: float = 3.0
    detectors: tuple[str, ...] = ("robust_threshold", "ewma", "cusum")


DEFAULT_CONFIG = MonitorConfig()


@dataclass
class MonitorResult:
    scores: pd.DataFrame
    alerts: list[Alert]
    summary: pd.DataFrame

    def alerts_frame(self) -> pd.DataFrame:
        return pd.DataFrame([a.to_dict() for a in self.alerts])


def score(
    measurements: pd.DataFrame, reference: ReferenceProfile, config: MonitorConfig
) -> pd.DataFrame:
    """Robust z and detector statistics for every valid observation."""
    data = valid_values(measurements)
    data = data[data["variable"].isin(VARIABLES.keys())]
    data = data.sort_values(["batch_id", "variable", "elapsed_h"]).copy()
    ref = reference.lookup(data["variable"], data["elapsed_h"])
    data["ref_median"] = ref["median"]
    data["ref_scale"] = ref["scale"]
    data["ref_n_batches"] = ref["n_batches"]
    data["z"] = (data["value"] - data["ref_median"]) / data["ref_scale"]
    # Where too few reference batches cover this hour (e.g. a batch running longer than
    # almost all approved batches), there is no meaningful "normal" to compare with:
    # no score, so no alert. The batch summary reports the coverage gap instead.
    data["beyond_reference"] = data["ref_n_batches"].fillna(0) < config.min_reference_batches
    data.loc[data["beyond_reference"], "z"] = np.nan

    stat_cols = {d: np.full(len(data), np.nan) for d in DETECTORS}
    positions = np.arange(len(data))
    for idx in data.groupby(["batch_id", "variable"], sort=False).indices.values():
        stats = statistics(data["z"].to_numpy()[idx], config.ewma_lambda, config.cusum_k)
        for d, values in stats.items():
            stat_cols[d][positions[idx]] = values
    for d, values in stat_cols.items():
        data[f"stat_{d}"] = values
    return data.reset_index(drop=True)


def loo_peak_statistics(
    measurements: pd.DataFrame, approved: list[str], config: MonitorConfig
) -> dict[str, dict[str, dict[str, float]]]:
    """Per detector and variable: each approved batch's peak |statistic| when scored
    against a reference built without it (batch id -> peak)."""
    peaks: dict[str, dict[str, dict[str, float]]] = {
        d: {v: {} for v in VARIABLES} for d in DETECTORS
    }
    for held_out in approved:
        ref = build_reference(measurements, [b for b in approved if b != held_out], config.window_h)
        s = score(measurements[measurements["batch_id"] == held_out], ref, config)
        for d in DETECTORS:
            batch_peaks = s.groupby("variable")[f"stat_{d}"].apply(
                lambda x: np.nanmax(np.abs(x)) if x.notna().any() else np.nan
            )
            for v, peak in batch_peaks.items():
                if np.isfinite(peak):
                    peaks[d][v][held_out] = float(peak)
    return peaks


def _limits(peaks, quantiles: dict[str, float], defaults: dict, margin: float) -> dict:
    return {
        d: {
            v: round(
                max(
                    defaults[d],
                    margin * float(np.quantile(list(peaks[d][v].values()), quantiles[d]))
                    if peaks[d][v]
                    else 0.0,
                ),
                4,
            )
            for v in VARIABLES
        }
        for d in DETECTORS
    }


def _alerting_batches(peaks, limits: dict, detectors) -> set[str]:
    return {
        b for d in detectors for v in VARIABLES for b, p in peaks[d][v].items() if p > limits[d][v]
    }


def fit_reference(
    measurements: pd.DataFrame, approved_batch_ids, config: MonitorConfig = DEFAULT_CONFIG
) -> ReferenceProfile:
    """Phase I: build the reference and calibrate frozen per-variable limits."""
    approved = sorted(set(approved_batch_ids))
    if len(approved) < config.min_reference_batches:
        raise ValueError(
            f"Need at least {config.min_reference_batches} approved batches, got {len(approved)}"
        )
    defaults = default_limits(
        config.ewma_lambda, config.ewma_L, config.robust_limit, config.cusum_h
    )
    quantiles = {
        d: config.threshold_calibration_quantile
        if d == "robust_threshold"
        else config.calibration_quantile
        for d in DETECTORS
    }
    if not all(0.5 <= q <= 1.0 for q in quantiles.values()):
        raise ValueError("calibration quantiles must be between 0.5 and 1.0")

    peaks = loo_peak_statistics(measurements, approved, config)
    margin = config.calibration_margin
    if config.false_alarm_budget is not None:
        # Tighten the robust-threshold limit step by step while the share of approved
        # batches that would have alerted (on any variable, any detector) stays within
        # the budget. The budget is a batch-level false-alarm rate, so it does not grow
        # with the number of monitored variables.
        allowed = config.false_alarm_budget * len(approved)
        for q in np.round(np.arange(quantiles["robust_threshold"], 0.795, -0.005), 3):
            trial = {**quantiles, "robust_threshold": float(q)}
            alerting = _alerting_batches(
                peaks, _limits(peaks, trial, defaults, margin), config.detectors
            )
            if len(alerting) > allowed:
                break
            quantiles = trial
    thresholds = _limits(peaks, quantiles, defaults, margin)
    alerting = _alerting_batches(peaks, thresholds, config.detectors)
    calibration = {
        "method": (
            "leave-one-out over approved batches; limit = max(default, margin * "
            "quantile of per-batch peak |statistic|)"
            + (
                f"; robust-threshold quantile chosen so at most {config.false_alarm_budget:.0%} "
                "of approved batches would alert"
                if config.false_alarm_budget is not None
                else ""
            )
        ),
        "quantiles": quantiles,
        "default_limits": {k: round(v, 4) for k, v in defaults.items()},
        "loo_max_statistic": {
            d: {v: round(max(p.values()), 4) if p else 0.0 for v, p in vs.items()}
            for d, vs in peaks.items()
        },
        # Approved batches that would have alerted under the frozen limits: the price
        # of tighter limits, recorded so reviewers can see it.
        "approved_batches_alerting": sorted(alerting),
        "config": asdict(config),
    }
    return build_reference(measurements, approved, config.window_h).with_thresholds(
        thresholds, calibration
    )


def _episodes(exceed: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of consecutive True runs."""
    runs, start = [], None
    for i, flag in enumerate(exceed):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(exceed) - 1))
    return runs


def monitor(
    measurements: pd.DataFrame,
    reference: ReferenceProfile,
    config: MonitorConfig = DEFAULT_CONFIG,
    as_of_h: float | None = None,
) -> MonitorResult:
    """Phase II: score batches against a frozen reference and raise structured alerts."""
    if not reference.thresholds:
        raise ValueError("Reference has no frozen thresholds; build it with fit_reference()")
    if as_of_h is not None:
        measurements = measurements[measurements["available_h"] <= as_of_h]
    scores = score(measurements, reference, config)

    alerts: list[Alert] = []
    for (batch_id, variable), g in scores.groupby(["batch_id", "variable"], sort=False):
        for d in config.detectors:
            stat = g[f"stat_{d}"].to_numpy()
            limit = reference.thresholds[d][variable]
            exceed = np.nan_to_num(np.abs(stat), nan=0.0) > limit
            for s, e in _episodes(exceed):
                window = g.iloc[s : e + 1]
                peak = int(np.nanargmax(np.abs(window[f"stat_{d}"].to_numpy())))
                row = window.iloc[peak]
                alerts.append(
                    Alert(
                        batch_id=batch_id,
                        variable=variable,
                        detector=d,
                        stage=""
                        if pd.isna(window.iloc[0]["stage"])
                        else str(window.iloc[0]["stage"]),
                        start_h=float(window.iloc[0]["elapsed_h"]),
                        end_h=float(window.iloc[-1]["elapsed_h"]),
                        first_raised_h=float(window.iloc[0]["available_h"]),
                        n_points=len(window),
                        persistence="isolated" if len(window) == 1 else "persistent",
                        direction="high" if row[f"stat_{d}"] > 0 else "low",
                        peak_statistic=round(float(row[f"stat_{d}"]), 3),
                        limit=limit,
                        observed_at_peak=float(row["value"]),
                        reference_median_at_peak=float(row["ref_median"]),
                        reference_scale_at_peak=float(row["ref_scale"]),
                        robust_z_at_peak=round(float(row["z"]), 2),
                        algorithm_version=ALGORITHM_VERSION,
                        reference_version=reference.version,
                    )
                )

    _add_context(alerts, measurements, scores, config)
    return MonitorResult(scores, alerts, summarise(alerts, measurements, scores))


def _add_context(
    alerts: list[Alert], measurements: pd.DataFrame, scores: pd.DataFrame, config: MonitorConfig
) -> None:
    """Corroborating variables and data-quality warnings near each alert."""
    w = config.corroboration_window_h
    by_batch: dict[str, list[Alert]] = {}
    for a in alerts:
        by_batch.setdefault(a.batch_id, []).append(a)
    flagged_rows = measurements[measurements["quality_flag"] != "valid"]

    for batch_id, batch_alerts in by_batch.items():
        bad = flagged_rows[flagged_rows["batch_id"] == batch_id]
        for a in batch_alerts:
            a.corroborating_variables = sorted(
                {
                    o.variable
                    for o in batch_alerts
                    if o.variable != a.variable
                    and o.start_h <= a.end_h + w
                    and o.end_h >= a.start_h - w
                }
            )
            near = bad[
                (bad["variable"] == a.variable)
                & bad["elapsed_h"].between(a.start_h - w, a.end_h + w)
            ]
            for flag, n in near.groupby("quality_flag").size().items():
                a.data_quality_warnings.append(f"{n} {flag} reading(s) within {w:g} h (excluded)")
            seg = scores[
                (scores["batch_id"] == batch_id)
                & (scores["variable"] == a.variable)
                & scores["elapsed_h"].between(a.start_h, a.end_h)
            ]
            if (seg["ref_n_batches"] < config.min_reference_batches).any():
                a.data_quality_warnings.append("reference built from few batches at this time")


def summarise(
    alerts: list[Alert], measurements: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    """One row per batch. Process deviation and contamination evidence are kept separate."""
    rows = []
    for batch_id in sorted(measurements["batch_id"].unique()):
        ba = [a for a in alerts if a.batch_id == batch_id]
        s = scores[scores["batch_id"] == batch_id]
        persistent_vars = sorted({a.variable for a in ba if a.persistence == "persistent"})
        if len(persistent_vars) >= 2:
            deviation = "substantially abnormal"
        elif ba:
            deviation = "abnormal"
        else:
            deviation = "within reference"
        first = min(ba, key=lambda a: a.first_raised_h) if ba else None
        m = measurements[measurements["batch_id"] == batch_id]
        rows.append(
            {
                "batch_id": batch_id,
                "process_deviation": deviation,
                "n_alerts": len(ba),
                "variables_flagged": ", ".join(sorted({a.variable for a in ba})),
                "persistent_variables": ", ".join(persistent_vars),
                "detectors_triggered": ", ".join(sorted({a.detector for a in ba})),
                "first_alert_raised_h": first.first_raised_h if first else np.nan,
                "first_alert_variable": first.variable if first else "",
                "max_abs_robust_z": round(float(np.nanmax(np.abs(s["z"]))), 2)
                if s["z"].notna().any()
                else np.nan,
                "non_valid_readings": int((m["quality_flag"] != "valid").sum()),
                "beyond_reference_from_h": float(s.loc[s["beyond_reference"], "elapsed_h"].min())
                if s["beyond_reference"].any()
                else np.nan,
                "contamination_evidence": NO_EVIDENCE,
            }
        )
    return pd.DataFrame(rows)
