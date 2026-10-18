"""Score the model on an external campaign (IndPenSim).

What is measured, on batches NOT used to build the reference:

* False alarms on normal batches, per control strategy.
* Fault batches: flagged at or after the fault onset, hours from onset to the first
  such alert, and whether that alert came during the first fault window (+12 h).
  Alerts before the onset are counted as false alerts.
* Diagnosis: the leading explanation per batch. The faults are equipment/actuator
  faults, so "Biological contamination" leading on any batch is a wrong call; a
  "raw-material or process-condition change" (H03) is the closest correct reading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluate import per_batch
from ..pipeline import fit, load_campaign, run_campaign
from .indpensim import reference_batches

CAVEAT = (
    "IndPenSim is a high-fidelity SIMULATION of an industrial penicillin process (not "
    "MaverickX, not contamination). These results test false alarms on realistic batch "
    "variation and the handling of real process faults."
)


def _first_window_end(windows: str) -> float:
    if not isinstance(windows, str) or not windows:
        return np.nan
    first = windows.split(";")[0].replace(" h", "").strip()
    return float(first.split("-")[1])


def evaluate_indpensim(campaign_dir: Path, out_dir: Path = Path("outputs/indpensim")) -> dict:
    campaign = load_campaign(campaign_dir, reference_ids=reference_batches())
    reference = fit(campaign)
    result = run_campaign(campaign, reference)
    alerts = result.diagnosis.monitor_result.alerts

    outcomes = campaign.outcomes.set_index("batch_id")
    batches = campaign.batches.set_index("batch_id")
    test = outcomes.loc[campaign.monitored_ids].reset_index()
    pb = per_batch(alerts, test).set_index("batch_id")
    table = (
        result.diagnosis.table().set_index("batch_id")
        if result.diagnosis.reports
        else pd.DataFrame()
    )

    rows = []
    for b in campaign.monitored_ids:
        o, row = outcomes.loc[b], pb.loc[b]
        lead = table["leading_id"].get(b, "") if len(table) else ""
        lead_name = table["leading_explanation"].get(b, "") if len(table) else ""
        win_end = _first_window_end(o.get("fault_windows", ""))
        rows.append(
            {
                "batch_id": b,
                "control": batches.loc[b, "control_strategy"],
                "scenario": o["scenario"],
                "category": o["category"],
                "flagged_after_onset": bool(row["flagged"]),
                "any_alert": bool(np.isfinite(row["first_alert_any_h"])),
                "n_alerts": int(row["n_alerts"]),
                "false_alerts": int(row["false_alerts"]),
                "onset_h": o["onset_h"],
                "first_alert_after_onset_h": row["first_alert_h"],
                "onset_to_alert_h": row["onset_to_alert_h"],
                "caught_in_first_window": bool(
                    np.isfinite(row["first_alert_h"]) and row["first_alert_h"] <= win_end + 12
                ),
                "variables_flagged": ", ".join(
                    sorted({a.variable for a in alerts if a.batch_id == b})
                ),
                "leading_id": lead or "(no alerts)",
                "leading_explanation": lead_name or "(no alerts)",
            }
        )
    df = pd.DataFrame(rows)
    normal, faults = df[df["category"] == "normal"], df[df["category"] != "normal"]

    by_control = (
        normal.groupby("control")
        .agg(
            batches=("batch_id", "size"),
            flagged=("any_alert", "sum"),
            alerts_per_batch=("n_alerts", "mean"),
        )
        .round(2)
    )
    fault_summary = {
        "fault_batches": len(faults),
        "flagged_after_onset": int(faults["flagged_after_onset"].sum()),
        "caught_in_first_window": int(faults["caught_in_first_window"].sum()),
        "median_onset_to_alert_h": float(np.nanmedian(faults["onset_to_alert_h"]))
        if faults["flagged_after_onset"].any()
        else np.nan,
        "false_alerts_before_onset": int(faults["false_alerts"].sum()),
    }
    leading_counts = (
        df.groupby(["category", "leading_explanation"]).size().rename("batches").reset_index()
    )
    contamination_leading = int((df["leading_id"] == "H01").sum())

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "batch_results.csv", index=False)
    result.diagnosis.monitor_result.alerts_frame().to_csv(out_dir / "alerts.csv", index=False)
    reference.save(out_dir / "reference_profile.json")
    for b, rep in result.diagnosis.reports.items():
        (out_dir / f"report_{b}.md").write_text(rep.to_markdown())

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(CAVEAT, "\n")
    print(
        f"Reference {reference.version}: {len(reference.batch_ids)} normal batches (20 per control strategy)\n"
    )
    print("Held-out NORMAL batches (false alarms):")
    print(by_control.to_string(), "\n")
    print("FAULT batches:", fault_summary)
    print(
        faults[
            [
                "batch_id",
                "scenario",
                "onset_h",
                "first_alert_after_onset_h",
                "onset_to_alert_h",
                "caught_in_first_window",
                "variables_flagged",
                "leading_explanation",
            ]
        ].to_string(index=False),
        "\n",
    )
    print("Leading explanation by category:")
    print(leading_counts.to_string(index=False))
    print(
        f"\nBatches where contamination was the leading explanation: {contamination_leading} (all would be wrong)"
    )
    return {
        "batches": df,
        "by_control": by_control,
        "faults": fault_summary,
        "leading": leading_counts,
        "reference": reference,
        "result": result,
        "campaign": campaign,
    }
