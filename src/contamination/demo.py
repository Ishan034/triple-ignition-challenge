"""End-to-end demonstration on synthetic data.

    python -m contamination.demo [--out outputs] [--seed 7]

Steps: generate synthetic batches -> write CSVs -> ingest and validate -> fit a frozen
reference on the earliest 30 (normal) batches -> monitor the later batches -> diagnose
every flagged batch -> evaluate against the separate outcomes table -> write alerts,
summaries, investigation reports and figures.

Diagnosis is evaluated on process evidence only (no recorded test results) at several
review times: at the first alert, 4 h and 8 h later, and at the end of the batch. The
written reports also include the simulated check results in investigations.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .diagnosis import diagnose
from .evaluate import (
    SYNTHETIC_CAVEAT,
    compare_detectors,
    evaluate_diagnosis,
    per_batch,
    review_anchor_times,
)
from .ingest import load_batches, load_investigations, load_measurements, load_outcomes
from .monitor import MonitorConfig, fit_reference, monitor
from .plots import plot_batch
from .synthetic import generate_dataset


def run(data_dir: Path, out_dir: Path, seed: int, n_reference: int = 30) -> dict:
    generate_dataset(seed=seed, n_reference=n_reference).save(data_dir)

    ingested = load_measurements(data_dir / "measurements.csv")
    outcomes = load_outcomes(data_dir / "outcomes.csv")
    batches = load_batches(data_dir / "batches.csv")
    investigations = load_investigations(data_dir / "investigations.csv")
    m = ingested.measurements

    # Chronological, whole-batch split. In real use the reference list is the set of
    # batches scientists approve as normal; here the earliest batches are normal by design.
    ordered = batches.sort_values("start_time")["batch_id"].tolist()
    reference_ids, test_ids = ordered[:n_reference], ordered[n_reference:]

    config = MonitorConfig()
    reference = fit_reference(m, reference_ids, config)
    result = monitor(m[m["batch_id"].isin(test_ids)], reference, config)

    test_outcomes = outcomes[outcomes["batch_id"].isin(test_ids)]
    comparison = compare_detectors(result.alerts, test_outcomes, config.detectors)
    batch_eval = per_batch(result.alerts, test_outcomes)

    # Diagnosis with the recorded (simulated) check results: the reports scientists see.
    test_m = m[m["batch_id"].isin(test_ids)]
    diagnosis = diagnose(test_m, reference, config, investigations)

    # Diagnosis evaluation on process evidence only, as of several review times.
    # Review times are anchored on the first alert at/after the simulated onset.
    first_alert = review_anchor_times(result.alerts, test_outcomes)
    review_tables = {}
    for label, delay in (
        ("at first alert", 0.0),
        ("first alert + 4 h", 4.0),
        ("first alert + 8 h", 8.0),
    ):
        tables = [
            diagnose(test_m[test_m["batch_id"] == b], reference, config, as_of_h=t + delay).table()
            for b, t in first_alert.items()
        ]
        review_tables[label] = pd.concat(tables, ignore_index=True)
    review_tables["end of batch"] = diagnose(test_m, reference, config).table()
    diagnosis_eval = evaluate_diagnosis(review_tables, test_outcomes)

    out_dir.mkdir(parents=True, exist_ok=True)
    reference.save(out_dir / "reference_profile.json")
    ingested.issues.to_csv(out_dir / "ingest_issues.csv", index=False)
    result.alerts_frame().to_csv(out_dir / "alerts.csv", index=False)
    result.summary.to_csv(out_dir / "batch_summary.csv", index=False)
    comparison.to_csv(out_dir / "detector_comparison.csv", index=False)
    batch_eval.to_csv(out_dir / "batch_evaluation.csv", index=False)
    diagnosis.table().to_csv(out_dir / "diagnosis.csv", index=False)
    diagnosis_eval.to_csv(out_dir / "diagnosis_evaluation.csv", index=False)
    report_dir = out_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    for batch_id, report in diagnosis.reports.items():
        (report_dir / f"{batch_id}.md").write_text(report.to_markdown())

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    for scenario in ("contamination", "slow_growth", "sensor_spike", "do_probe_drift", "normal"):
        row = test_outcomes[test_outcomes["scenario"] == scenario].iloc[0]
        markers = {}
        if pd.notna(row["onset_h"]):
            markers["simulated onset"] = row["onset_h"]
        if scenario == "contamination" and pd.notna(row["reference_event_h"]):
            markers["visual/odour note"] = row["reference_event_h"]
        plot_batch(
            result.scores,
            result.alerts,
            reference,
            row["batch_id"],
            fig_dir / f"{row['batch_id']}_{scenario}.png",
            markers=markers,
            title_note=f" ({scenario}, synthetic)",
        )
    return {
        "reference": reference,
        "result": result,
        "comparison": comparison,
        "batch_eval": batch_eval,
        "issues": ingested.issues,
        "test_outcomes": test_outcomes,
        "diagnosis": diagnosis,
        "diagnosis_eval": diagnosis_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    r = run(args.data, args.out, args.seed)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(SYNTHETIC_CAVEAT, "\n")
    print(
        f"Reference version {r['reference'].version} built from {len(r['reference'].batch_ids)} batches"
    )
    print(f"Ingest issues: {len(r['issues'])}\n")
    print("Detector comparison (test batches):")
    print(r["comparison"].round(2).to_string(index=False), "\n")
    print("Diagnosis on process evidence only (leading explanation vs simulated scenario):")
    print(r["diagnosis_eval"].to_string(index=False), "\n")
    table = (
        r["diagnosis"].table().merge(r["test_outcomes"][["batch_id", "scenario"]], on="batch_id")
    )
    print("Investigation summary (including recorded check results):")
    cols = [
        "batch_id",
        "scenario",
        "leading_explanation",
        "contamination_evidence",
        "confidence",
        "next_check",
    ]
    print(table[cols].sort_values("scenario").to_string(index=False))
    print(f"\nOutputs written to {args.out}/")


if __name__ == "__main__":
    main()
