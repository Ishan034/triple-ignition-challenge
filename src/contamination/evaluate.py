"""Batch-level evaluation of detectors against reviewed outcomes.

The outcomes table is used *only* here, never by the reference or detectors.

* A contaminated or non-contamination-anomaly batch counts as flagged only if an
  alert was raised at or after the onset; alerts before the onset cannot be caused
  by it and are counted as false alerts instead.
* ``lead_time_vs_reference_h`` = reference event time (when today's method, the
  visual/odour note, would have caught it) minus first alert time. Positive means the
  system raised the alert earlier.
* With few cases these estimates are very uncertain: report counts, not just ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .alerts import Alert

SYNTHETIC_CAVEAT = (
    "SYNTHETIC DATA: these figures show the software behaves as intended on simulated "
    "batches. They are not real-world contamination-detection performance."
)


def _first_alert(alerts: list[Alert], batch_id: str, after_h: float | None) -> float:
    times = [
        a.first_raised_h
        for a in alerts
        if a.batch_id == batch_id and (after_h is None or a.start_h >= after_h)
    ]
    return min(times) if times else np.nan


def per_batch(
    alerts: list[Alert], outcomes: pd.DataFrame, detectors: set[str] | None = None
) -> pd.DataFrame:
    selected = [a for a in alerts if detectors is None or a.detector in detectors]
    rows = []
    for o in outcomes.itertuples(index=False):
        onset = None if pd.isna(o.onset_h) else float(o.onset_h)
        first_any = _first_alert(selected, o.batch_id, None)
        first_after = _first_alert(selected, o.batch_id, onset) if onset is not None else first_any
        n_alerts = sum(a.batch_id == o.batch_id for a in selected)
        n_before = sum(
            a.batch_id == o.batch_id and onset is not None and a.start_h < onset for a in selected
        )
        rows.append(
            {
                "batch_id": o.batch_id,
                "category": o.category,
                "scenario": o.scenario,
                "flagged": bool(np.isfinite(first_after)),
                "first_alert_h": first_after,
                "n_alerts": n_alerts,
                "false_alerts": n_alerts if o.category == "normal" else n_before,
                "onset_to_alert_h": first_after - onset if onset is not None else np.nan,
                "lead_time_vs_reference_h": (o.reference_event_h - first_after)
                if o.category == "contamination"
                else np.nan,
                "first_alert_any_h": first_any,
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    alerts: list[Alert], outcomes: pd.DataFrame, detectors: set[str] | None = None
) -> dict:
    pb = per_batch(alerts, outcomes, detectors)
    cont = pb[pb["category"] == "contamination"]
    normal = pb[pb["category"] == "normal"]
    other = pb[pb["category"] == "non_contamination_anomaly"]
    flagged = pb[pb["flagged"] | (pb["first_alert_any_h"].notna())]
    n_flagged_cont = int(cont["flagged"].sum())
    return {
        "detectors": "all" if detectors is None else "+".join(sorted(detectors)),
        "contamination_flagged": f"{n_flagged_cont}/{len(cont)}",
        "sensitivity": n_flagged_cont / len(cont) if len(cont) else np.nan,
        "normal_flagged": f"{int(normal['flagged'].sum())}/{len(normal)}",
        "specificity": 1 - normal["flagged"].mean() if len(normal) else np.nan,
        "non_contam_anomaly_flagged": f"{int(other['flagged'].sum())}/{len(other)}",
        "precision_for_contamination": n_flagged_cont / len(flagged) if len(flagged) else np.nan,
        "false_alerts_per_batch": pb["false_alerts"].sum() / len(pb) if len(pb) else np.nan,
        "median_onset_to_alert_h": float(np.nanmedian(cont["onset_to_alert_h"]))
        if n_flagged_cont
        else np.nan,
        "median_lead_vs_reference_h": float(np.nanmedian(cont["lead_time_vs_reference_h"]))
        if n_flagged_cont
        else np.nan,
    }


def compare_detectors(alerts: list[Alert], outcomes: pd.DataFrame, detector_names) -> pd.DataFrame:
    rows = [evaluate(alerts, outcomes, {d}) for d in detector_names]
    rows.append(evaluate(alerts, outcomes, None))
    return pd.DataFrame(rows)


# Expected leading explanation for each synthetic scenario (diagnosis evaluation only).
EXPECTED_HYPOTHESIS = {
    "contamination": "H01",
    "slow_growth": "H02",
    "sensor_spike": "H04",
    "do_probe_drift": "H04",
}


def evaluate_diagnosis(
    diagnosis_tables: dict[str, pd.DataFrame], outcomes: pd.DataFrame
) -> pd.DataFrame:
    """Compare the leading explanation with the known scenario at several review times.

    ``diagnosis_tables`` maps a label (e.g. "at first alert", "end of batch") to a
    DiagnosisResult.table(). "unresolved" (no explanation strong enough to lead) is the
    safe failure: the scientist is asked to investigate more broadly. "wrong_leading"
    is the harmful one: it points the investigation the wrong way.
    """
    rows = []
    for label, table in diagnosis_tables.items():
        merged = outcomes.merge(table, on="batch_id", how="inner")
        merged = merged[merged["scenario"].isin(EXPECTED_HYPOTHESIS)]
        for scenario, g in merged.groupby("scenario"):
            expected = EXPECTED_HYPOTHESIS[scenario]
            rows.append(
                {
                    "review_time": label,
                    "scenario": scenario,
                    "expected": expected,
                    "batches": len(g),
                    "matched": int((g["leading_id"] == expected).sum()),
                    "unresolved": int(g["leading_id"].isin(["", "H05"]).sum()),
                    "wrong_leading": int((~g["leading_id"].isin([expected, "", "H05"])).sum()),
                }
            )
    return pd.DataFrame(rows)


def review_anchor_times(alerts: list[Alert], outcomes: pd.DataFrame) -> dict[str, float]:
    """First alert time per batch to anchor review times on.

    For batches with a known onset this is the first alert raised at or after the onset;
    earlier alerts are unrelated noise and are evaluated as false alerts instead.
    """
    out = {}
    for o in outcomes.itertuples(index=False):
        onset = None if pd.isna(o.onset_h) else float(o.onset_h)
        t = _first_alert(alerts, o.batch_id, onset)
        if np.isfinite(t):
            out[o.batch_id] = t
    return out


def evaluate_tracing(trace, outcomes: pd.DataFrame) -> dict:
    """Compare source-tracing leads with the planted (simulated) common source."""
    src = outcomes["simulated_source"].fillna("").astype(str)
    planted_names = sorted({s for s in src if s and not s.startswith("independent")})
    planted = planted_names[0] if planted_names else None
    planted_cases = outcomes.loc[src == planted, "batch_id"].tolist() if planted else []
    independent = outcomes.loc[src.str.startswith("independent"), "batch_id"].tolist()
    tiers = {c.resource: c.tier for c in trace.candidates}
    strong = [c.resource for c in trace.candidates if c.tier == "strong lead"]
    leads = trace.leads

    def linked_to(batch_id: str, resource: str) -> bool:
        return any(s.resource == resource for s in trace.per_case.get(batch_id, []))

    def strong_link(batch_id: str) -> bool:
        return any(s.tier == "strong lead" for s in trace.per_case.get(batch_id, []))

    return {
        "planted_source": planted or "",
        "planted_tier": tiers.get(planted, "not a candidate") if planted else "",
        "planted_is_top_lead": bool(planted and leads and leads[0].resource == planted),
        "false_strong_leads": ", ".join(s for s in strong if s != planted),
        "n_possible_leads": sum(c.tier == "possible lead" for c in trace.candidates),
        "planted_cases_linked": f"{sum(linked_to(b, planted) for b in planted_cases)}/{len(planted_cases)}",
        "independent_cases_without_strong_lead": f"{sum(not strong_link(b) for b in independent)}/{len(independent)}",
    }
