"""Reference refresh: propose newly approved batches, show the drift, keep a history.

A frozen reference ages: raw materials, equipment and recipes change, and on the
erythromycin plant the later batches drifted from the earlier ones. This module
supports the periodic review that keeps "normal" current, without ever approving a
batch automatically:

1. **Propose** candidates: monitored batches that stayed within the reference (no
   alerts) and have no contamination evidence. A scientist confirms each one; batches
   with a recorded problem outcome are never proposed.
2. **Compare** the current reference with the one the refresh would produce: per
   variable, how far the median trajectory moves (in units of the current reference's
   robust spread) and how the alert limits change. A large shift is a finding in its
   own right (a process change to explain), not just a reason to refresh.
3. **Record** every applied refresh in ``reference_history.csv`` (version, date,
   batches added and removed, reviewer note) next to ``reference_batches.txt``.

    python -m contamination.refresh data/erythromycin                  # propose + compare
    python -m contamination.refresh data/erythromycin --apply --note "Q3 review, AB"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import REFERENCE_LIST, Campaign, CampaignResult, fit, load_campaign, run_campaign
from .reference import ReferenceProfile

HISTORY = "reference_history.csv"
HISTORY_COLUMNS = ["version", "applied_at", "n_batches", "added", "removed", "note"]


def propose(campaign: Campaign, result: CampaignResult) -> pd.DataFrame:
    """Monitored batches that could join the reference, with why or why not."""
    summary = result.diagnosis.monitor_result.summary.set_index("batch_id")
    outcomes = campaign.outcomes.set_index("batch_id") if campaign.outcomes is not None else None
    rows = []
    for b in campaign.monitored_ids:
        n_alerts = int(summary["n_alerts"].get(b, 0))
        report = result.diagnosis.reports.get(b)
        evidence = report.assessment["contamination_evidence"] if report else ""
        category = outcomes["category"].get(b, "") if outcomes is not None else ""
        if category and category != "normal":
            ok, why = False, f"recorded outcome: {category}"
        elif evidence.startswith("confirmed"):
            ok, why = False, "contamination confirmed by test"
        elif n_alerts:
            ok, why = False, f"{n_alerts} alert(s): review before approving"
        else:
            ok, why = True, "stayed within the reference"
        rows.append({"batch_id": b, "candidate": ok, "reason": why})
    return pd.DataFrame(rows)


def refreshed_ids(
    current: list[str], add: list[str], ordered: list[str], keep_latest: int | None
) -> list[str]:
    """Current approved batches plus ``add``; optionally only the latest ``keep_latest``
    by start time (a rolling window, so old operating conditions age out)."""
    ids = [b for b in ordered if b in set(current) | set(add)]
    return ids[-keep_latest:] if keep_latest else ids


def drift(old: ReferenceProfile, new: ReferenceProfile) -> pd.DataFrame:
    """Per variable: median shift of the reference trajectory in units of the old robust
    spread, the largest hourly shift, and how the single-reading limit changes."""
    a = old.stats.set_index(["variable", "hour"])
    b = new.stats.set_index(["variable", "hour"])
    both = a[["median", "scale"]].join(b[["median"]], rsuffix="_new", how="inner")
    both["shift"] = (both["median_new"] - both["median"]) / both["scale"]
    rows = []
    for var, g in both.groupby(level="variable"):
        s = g["shift"].to_numpy()
        t_old = old.thresholds.get("robust_threshold", {}).get(var, np.nan)
        t_new = new.thresholds.get("robust_threshold", {}).get(var, np.nan)
        rows.append(
            {
                "variable": var,
                "median_shift_sd": round(float(np.median(s)), 2),
                "largest_shift_sd": round(float(s[np.argmax(np.abs(s))]), 2),
                "hours_compared": len(s),
                "limit_old": t_old,
                "limit_new": t_new,
                "flag": "review" if abs(np.median(s)) >= 1.0 else "",
            }
        )
    return pd.DataFrame(rows).sort_values("median_shift_sd", key=np.abs, ascending=False)


@dataclass
class RefreshPlan:
    candidates: pd.DataFrame
    new_ids: list[str]
    added: list[str]
    removed: list[str]
    old: ReferenceProfile
    new: ReferenceProfile
    drift: pd.DataFrame


def plan(campaign_dir: str | Path, keep_latest: int | None = None) -> RefreshPlan:
    campaign = load_campaign(campaign_dir)
    old = fit(campaign)
    result = run_campaign(campaign, old)
    cands = propose(campaign, result)
    add = cands.loc[cands["candidate"], "batch_id"].tolist()
    ordered = campaign.batches.sort_values("start_time")["batch_id"].tolist()
    new_ids = refreshed_ids(campaign.reference_ids, add, ordered, keep_latest)
    refreshed = Campaign(**{**campaign.__dict__, "reference_ids": new_ids})
    new = fit(refreshed)
    return RefreshPlan(
        cands,
        new_ids,
        sorted(set(new_ids) - set(campaign.reference_ids)),
        sorted(set(campaign.reference_ids) - set(new_ids)),
        old,
        new,
        drift(old, new),
    )


def apply(campaign_dir: str | Path, p: RefreshPlan, note: str) -> Path:
    """Write the new approved list and append to the history. Old list kept in history."""
    d = Path(campaign_dir)
    (d / REFERENCE_LIST).write_text("\n".join(p.new_ids) + "\n")
    path = d / HISTORY
    history = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=HISTORY_COLUMNS)
    if history.empty:  # record the starting point first
        history.loc[0] = [p.old.version, "", len(p.old.batch_ids), "", "", "initial reference"]
    history.loc[len(history)] = [
        p.new.version,
        datetime.now(UTC).isoformat(timespec="seconds"),
        len(p.new_ids),
        " ".join(p.added),
        " ".join(p.removed),
        note,
    ]
    history.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--keep-latest", type=int, default=None, help="rolling window size")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    p = plan(args.campaign, args.keep_latest)
    c = p.candidates
    print(f"Current reference {p.old.version}: {len(p.old.batch_ids)} batches")
    print(f"Candidates to approve: {int(c['candidate'].sum())} of {len(c)} monitored batches")
    print(c.loc[~c["candidate"], "reason"].value_counts().to_string(), "\n")
    print(
        f"Refreshed reference {p.new.version}: {len(p.new_ids)} batches "
        f"(+{len(p.added)}, -{len(p.removed)})\n"
    )
    print("How 'normal' would move (shift in units of the current reference's spread):")
    print(p.drift.to_string(index=False))
    if args.apply:
        if not args.note:
            raise SystemExit("--apply needs --note (who reviewed the candidates, and why)")
        print("\nApplied; history in", apply(args.campaign, p, args.note))
    else:
        print("\nNothing changed. Review the candidates, then re-run with --apply --note '...'.")


if __name__ == "__main__":
    main()
