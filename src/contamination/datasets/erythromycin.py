"""Erythromycin fermentation process (EFP): 406 real industrial batches.

Source: Zenodo record 14619074 (CC BY 4.0): historical production data from an
erythromycin plant in 2022, logged hourly, 23 process variables plus chemical potency.
There are NO fault or contamination labels, so this dataset answers a different
question from IndPenSim: on genuine plant noise, how many batches does the model
flag, and are the flagged batches actually worse? Final potency is held out of
monitoring and used only as the outcome to compare flagged and unflagged batches.

    python -m contamination.datasets.erythromycin prepare
    python -m contamination.datasets.erythromycin evaluate

Column codes are undocumented. Only signals whose meaning is clear are monitored:
te (temperature), phzx / phlx (online / offline pH), doz (dissolved oxygen),
zs (agitation), our and cer. Cumulative feeds are inputs, rq and kla are derived
ratios, and jn, nd, ht, zt, nn are left out until their meaning is confirmed.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..diagnosis import PROFILES_DIR
from ..schema import MEASUREMENT_COLUMNS, VARIABLES

URL = "https://zenodo.org/records/14619074/files/EFP_long.csv?download=1"
RAW_DIR = Path("data/external/erythromycin")
COLUMNS = {
    "te": "temperature",
    "phzx": "ph",
    "phlx": "ph_offline",
    "doz": "dissolved_oxygen",
    "zs": "agitation",
    "our": "our_efp",
    "cer": "cer_efp",
}
REFERENCE_FRACTION = 0.6


def download(raw_dir: Path = RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "EFP_long.csv"
    if not target.exists():
        with urllib.request.urlopen(URL) as r, open(target, "wb") as f:
            shutil.copyfileobj(r, f)
    return target


def convert(raw_csv: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    d = pd.read_csv(raw_csv, parse_dates=["date"])
    missing = set(COLUMNS) | {"date", "batch_id", "hh", "hx"}
    if missing - set(d.columns):
        raise ValueError(f"EFP columns not found: {sorted(missing - set(d.columns))}")
    d["batch"] = d["batch_id"].map(lambda b: f"E{int(b):03d}")
    # Batch start = first log time minus the elapsed hours already logged (hh).
    first = d.sort_values("date").groupby("batch").first()
    start = (first["date"] - pd.to_timedelta(first["hh"], unit="h")).rename("start_time")

    long = d.melt(
        id_vars=["batch", "date"], value_vars=list(COLUMNS), var_name="code", value_name="value"
    )
    long["variable"] = long["code"].map(COLUMNS)
    m = pd.DataFrame(
        {
            "batch_id": long["batch"],
            "timestamp": long["date"],
            "available_at": long["date"],
            "stage": np.nan,
            "variable": long["variable"],
            "value": long["value"].astype(float),
            "unit": long["variable"].map({k: v.unit for k, v in VARIABLES.items()}),
            "source": np.where(long["code"] == "phlx", "offline_assay", "plant_historian"),
            "quality_flag": "valid",
            "observation": None,
        }
    )[MEASUREMENT_COLUMNS]

    final_potency = d.sort_values("date").groupby("batch")["hx"].last()
    batches = pd.DataFrame(
        {
            "batch_id": start.index,
            "start_time": start.values,
            "bioreactor_id": "not recorded",
            "medium_lot": "not recorded",
            "inoculum_lot": "not recorded",
            "logged_hours": d.groupby("batch")["hh"].max().reindex(start.index).values,
        }
    )
    outcomes = pd.DataFrame(
        {
            "batch_id": final_potency.index,
            "category": "normal",  # unknown: no labels; treated as routine production
            "scenario": "production_batch",
            "outcome_status": "unknown",
            "onset_h": np.nan,
            "reference_event_h": np.nan,
            "final_potency": final_potency.values,
            "reason": "Real plant batch; no fault or contamination label available",
            "data_origin": "Erythromycin fermentation process, Zenodo 14619074, CC BY 4.0",
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"measurements": m, "batches": batches, "outcomes": outcomes}
    for name, df in result.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    shutil.copy(PROFILES_DIR / "erythromycin.yaml", out_dir / "profile.yaml")
    return result


def reference_batches(batches: pd.DataFrame) -> list[str]:
    """Chronological split: the earliest 60% of batches form the reference."""
    ordered = batches.sort_values("start_time")["batch_id"].tolist()
    return ordered[: int(len(ordered) * REFERENCE_FRACTION)]


def prepare(raw_dir: Path = RAW_DIR) -> Path:
    convert(download(raw_dir), raw_dir / "campaign")
    return raw_dir / "campaign"


def evaluate(campaign_dir: Path, out_dir: Path = Path("outputs/erythromycin")) -> dict:
    from ..pipeline import fit, load_campaign, run_campaign

    batches = pd.read_csv(campaign_dir / "batches.csv", parse_dates=["start_time"])
    campaign = load_campaign(campaign_dir, reference_ids=reference_batches(batches))
    reference = fit(campaign)
    result = run_campaign(campaign, reference)
    summary = result.diagnosis.monitor_result.summary.set_index("batch_id")
    outcomes = campaign.outcomes.set_index("batch_id")
    table = (
        result.diagnosis.table().set_index("batch_id")
        if result.diagnosis.reports
        else pd.DataFrame()
    )

    df = pd.DataFrame(
        {
            "batch_id": campaign.monitored_ids,
            "flagged": [
                summary.loc[b, "n_alerts"] > 0 if b in summary.index else False
                for b in campaign.monitored_ids
            ],
            "n_alerts": [
                summary.loc[b, "n_alerts"] if b in summary.index else 0
                for b in campaign.monitored_ids
            ],
            "process_deviation": [
                summary.loc[b, "process_deviation"] if b in summary.index else "no data"
                for b in campaign.monitored_ids
            ],
            "variables_flagged": [
                summary.loc[b, "variables_flagged"] if b in summary.index else ""
                for b in campaign.monitored_ids
            ],
            "non_valid_readings": [
                summary.loc[b, "non_valid_readings"] if b in summary.index else 0
                for b in campaign.monitored_ids
            ],
            "final_potency": [outcomes.loc[b, "final_potency"] for b in campaign.monitored_ids],
            "leading_explanation": [
                table["leading_explanation"].get(b, "(no alerts)") if len(table) else "(no alerts)"
                for b in campaign.monitored_ids
            ],
        }
    )
    low_cut = outcomes.loc[campaign.reference_ids, "final_potency"].quantile(0.10)
    df["low_potency"] = df["final_potency"] < low_cut
    rate = lambda s: f"{int(s.sum())}/{len(s)} ({s.mean():.0%})" if len(s) else "0/0"

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "batch_results.csv", index=False)
    result.diagnosis.monitor_result.alerts_frame().to_csv(out_dir / "alerts.csv", index=False)
    campaign.ingest.issues.to_csv(out_dir / "ingest_issues.csv", index=False)

    print("REAL plant data, no labels: flags are 'batches worth review', not confirmed faults.\n")
    print(
        f"Reference {reference.version}: {len(campaign.reference_ids)} earliest batches; {len(df)} later batches monitored"
    )
    print(
        f"Ingest: {len(campaign.ingest.issues)} issues flagged ->",
        campaign.ingest.summary().to_dict(),
        "\n",
    )
    print("Batches flagged:", rate(df["flagged"]))
    print("  substantially abnormal:", rate(df["process_deviation"] == "substantially abnormal"))
    print(
        f"Low-potency batches (final potency < reference 10th percentile, {low_cut:.0f}):",
        int(df["low_potency"].sum()),
    )
    print("  flagged among low-potency batches:", rate(df.loc[df["low_potency"], "flagged"]))
    print("  flagged among the rest:           ", rate(df.loc[~df["low_potency"], "flagged"]))
    print(
        "Median final potency: flagged",
        df.loc[df.flagged, "final_potency"].median(),
        "| not flagged",
        df.loc[~df.flagged, "final_potency"].median(),
    )
    print("\nLeading explanations among flagged batches:")
    print(df[df.flagged]["leading_explanation"].value_counts().to_string())
    print("\nVariables most often flagged:")
    print(
        pd.Series([v for s in df["variables_flagged"] for v in s.split(", ") if v])
        .value_counts()
        .to_string()
    )
    return {
        "batches": df,
        "low_cut": low_cut,
        "reference": reference,
        "result": result,
        "campaign": campaign,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["prepare", "evaluate"])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    if args.command == "prepare":
        print("campaign written to", prepare(args.raw_dir))
    else:
        evaluate(args.raw_dir / "campaign")


if __name__ == "__main__":
    main()
