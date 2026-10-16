"""Multi-seed robustness evaluation on synthetic data.

    python -m contamination.robustness --seeds 11 31 [--out outputs/robustness]

For each seed: generate a fresh synthetic dataset, fit a reference on the 30 earliest
(normal) batches, monitor and diagnose the later batches, and evaluate detection and
diagnosis. Diagnosis uses process evidence only (no recorded test results) at
several review times anchored on the first alert at or after the simulated onset.

Thresholds and knowledge-base weights were adjusted while looking at seeds 1-10,
so report results on other seeds. SYNTHETIC DATA: these results show software
behaviour, not real-world performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .diagnosis import diagnose
from .evaluate import SYNTHETIC_CAVEAT, evaluate, evaluate_diagnosis, review_anchor_times
from .ingest import validate_measurements
from .monitor import DEFAULT_CONFIG, fit_reference
from .synthetic import generate_dataset

REVIEW_DELAYS = (("at first alert", 0.0), ("first alert + 4 h", 4.0), ("first alert + 8 h", 8.0))


def run_seed(seed: int, n_reference: int = 30) -> tuple[dict, pd.DataFrame]:
    ds = generate_dataset(seed=seed, n_reference=n_reference)
    m = validate_measurements(ds.measurements).measurements
    ordered = ds.batches.sort_values("start_time")["batch_id"].tolist()
    reference = fit_reference(m, ordered[:n_reference], DEFAULT_CONFIG)
    test = m[m["batch_id"].isin(ordered[n_reference:])]
    outcomes = ds.outcomes[ds.outcomes["batch_id"].isin(ordered[n_reference:])]

    full = diagnose(test, reference, DEFAULT_CONFIG)
    detection = evaluate(full.monitor_result.alerts, outcomes)
    anchors = review_anchor_times(full.monitor_result.alerts, outcomes)
    tables = {}
    for label, delay in REVIEW_DELAYS:
        parts = [
            diagnose(
                test[test["batch_id"] == b], reference, DEFAULT_CONFIG, as_of_h=t + delay
            ).table()
            for b, t in anchors.items()
        ]
        tables[label] = pd.concat(parts, ignore_index=True)
    tables["end of batch"] = full.table()
    diag = evaluate_diagnosis(tables, outcomes)
    diag["seed"] = seed
    return {"seed": seed, **detection}, diag


def run(seeds) -> tuple[pd.DataFrame, pd.DataFrame]:
    detection, diagnosis = [], []
    for seed in seeds:
        d, g = run_seed(seed)
        detection.append(d)
        diagnosis.append(g)
    diag = pd.concat(diagnosis, ignore_index=True)
    totals = (
        diag.groupby(["review_time", "scenario"], sort=False)[
            ["batches", "matched", "unresolved", "wrong_leading"]
        ]
        .sum()
        .reset_index()
    )
    return pd.DataFrame(detection), totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, nargs=2, default=(11, 31), metavar=("FIRST", "STOP"))
    parser.add_argument("--out", type=Path, default=Path("outputs/robustness"))
    args = parser.parse_args()

    detection, totals = run(range(*args.seeds))
    args.out.mkdir(parents=True, exist_ok=True)
    detection.to_csv(args.out / "detection_by_seed.csv", index=False)
    totals.to_csv(args.out / "diagnosis_totals.csv", index=False)

    pd.set_option("display.width", 200)
    print(SYNTHETIC_CAVEAT, "\n")
    print(f"Seeds {args.seeds[0]}-{args.seeds[1] - 1}: detection per seed")
    cols = [
        "seed",
        "contamination_flagged",
        "normal_flagged",
        "non_contam_anomaly_flagged",
        "median_onset_to_alert_h",
        "median_lead_vs_reference_h",
    ]
    print(detection[cols].to_string(index=False), "\n")
    print("Diagnosis (process evidence only), totals across seeds:")
    print(totals.to_string(index=False))


if __name__ == "__main__":
    main()
