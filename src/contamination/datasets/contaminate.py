"""Semi-synthetic contamination test on IndPenSim batches. SIMULATED CONTAMINATION.

Our own simulator writes both the normal batches and the contaminant, so a perfect
score there partly reflects shared assumptions. Here the background is IndPenSim's
industrial-scale process: its batch-to-batch variation, three control strategies,
hourly online signals and 12-hourly offline assays with a 4 h result delay. Onto
copies of the held-out normal batches we add the effects of a contaminant that grows
from a random onset hour. The effects are still OUR ASSUMPTIONS (a plausible bacterial
contaminant in an aerobic fed-batch), so results are a software test on realistic
background variation, not real-world contamination-detection performance.

Contaminant growth is logistic, g(t) = 1 / (1 + exp(-(t - onset - 12) / 3)): about 2%
of full effect at onset, half at onset + 12 h. Effects scale with ``severity`` (0.5, 1,
2 and 4 are tested, to find how large a contaminant must be before it is caught). At
``severity`` 1:

* oxygen demand: OUR +20%; dissolved oxygen -25%
* CO2: CER and off-gas CO2 +20%
* biomass (offline): +15% (extra cells)
* penicillin (offline): -15% of the amount made after onset (competition for substrate)
* acid production: base flow + 60% of the batch's median base flow; pH dips by at most
  0.03 because the controller compensates
* heat: cooling-water flow + 30% of the batch's median; temperature +0.05 degC
* substrate: -30% (consumed by the contaminant)

    python -m contamination.datasets.contaminate          # build and evaluate
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..monitor import DEFAULT_CONFIG, MonitorConfig
from ..uncertainty import median_text, rate

SOURCE = Path("data/indpensim")
OUT = Path("outputs/indpensim_contaminated")
SEVERITIES = (0.5, 1.0, 2.0, 4.0)

MULTIPLY = {  # variable -> relative change at full effect
    "our": 0.20,
    "do2_mg_l": -0.25,
    "cer": 0.20,
    "co2_offgas": 0.20,
    "biomass_offline": 0.15,
    "substrate": -0.30,
}
ADD_TO_MEDIAN = {"base_flow": 0.60, "coolant_flow": 0.30}  # share of the batch's median
ADD = {"ph": -0.03, "temperature": 0.05}


def growth(t: np.ndarray, onset: float) -> np.ndarray:
    return np.where(t >= onset, 1.0 / (1.0 + np.exp(-(t - onset - 12.0) / 3.0)), 0.0)


def inject(batch: pd.DataFrame, onset: float, severity: float) -> pd.DataFrame:
    """A contaminated copy of one batch's measurements (same timestamps and delays)."""
    out = batch.copy()
    g = severity * growth(out["elapsed_h"].to_numpy(float), onset)
    v = out["value"].to_numpy(float).copy()
    var = out["variable"].to_numpy()
    for name, frac in MULTIPLY.items():
        sel = var == name
        v[sel] = v[sel] * (1.0 + frac * g[sel])
    for name, frac in ADD_TO_MEDIAN.items():
        sel = var == name
        if sel.any():
            v[sel] = v[sel] + frac * np.nanmedian(v[sel]) * g[sel]
    for name, delta in ADD.items():
        sel = var == name
        v[sel] = v[sel] + delta * g[sel]
    sel = var == "penicillin_offline"
    if sel.any():
        t = out["elapsed_h"].to_numpy(float)[sel]
        before = np.interp(onset, t, v[sel]) if (t <= onset).any() else 0.0
        made_after = np.clip(v[sel] - before, 0.0, None)
        v[sel] = v[sel] - 0.15 * g[sel] * made_after
    out["value"] = v
    return out


def build(source: Path = SOURCE, seed: int = 0):
    """Reference batches and held-out normal batches as they are, plus a contaminated
    copy of every held-out normal batch at each severity."""
    from ..pipeline import load_campaign

    camp = load_campaign(source)
    m = camp.measurements
    outcomes = camp.outcomes.set_index("batch_id")
    normal_test = [b for b in camp.monitored_ids if outcomes.loc[b, "category"] == "normal"]
    rng = np.random.default_rng(seed)
    copies, rows = [], []
    for b in normal_test:
        batch = m[m["batch_id"] == b]
        length = float(batch["elapsed_h"].max())
        for sev in SEVERITIES:
            onset = float(rng.integers(30, int(length) - 60))
            cid = f"{b}-C{int(sev * 100)}"
            copies.append(inject(batch, onset, sev).assign(batch_id=cid))
            rows.append(
                {
                    "batch_id": cid,
                    "original": b,
                    "category": "contamination",
                    "severity": sev,
                    "onset_h": onset,
                }
            )
    keep = m[m["batch_id"].isin(camp.reference_ids + normal_test)]
    data = pd.concat([keep, *copies], ignore_index=True)
    truth = pd.concat(
        [
            pd.DataFrame(
                {
                    "batch_id": normal_test,
                    "original": normal_test,
                    "category": "normal",
                    "severity": 0.0,
                    "onset_h": np.nan,
                }
            ),
            pd.DataFrame(rows),
        ],
        ignore_index=True,
    )
    return camp, data, truth


def evaluate(
    source: Path = SOURCE,
    out_dir: Path = OUT,
    seed: int = 0,
    config: MonitorConfig = DEFAULT_CONFIG,
) -> dict:
    from ..diagnosis import diagnose
    from ..pipeline import fit

    camp, data, truth = build(source, seed)
    reference = fit(camp, config)  # fitted on the untouched reference batches only
    test = data[data["batch_id"].isin(truth["batch_id"])]
    full = diagnose(test, reference, config, kb=camp.kb)
    alerts = full.monitor_result.alerts
    table = full.table().set_index("batch_id") if full.reports else pd.DataFrame()

    rows = []
    for t in truth.itertuples(index=False):
        onset = t.onset_h if pd.notna(t.onset_h) else -np.inf
        mine = [a for a in alerts if a.batch_id == t.batch_id]
        after = [a.first_raised_h for a in mine if a.start_h >= onset]
        first = min(after) if after else np.nan
        lead_named = np.nan
        if t.category == "contamination" and after:
            # When did contamination first become the leading explanation?
            batch = test[test["batch_id"] == t.batch_id]
            for h in np.arange(first, float(batch["available_h"].max()) + 4, 4.0):
                rep = diagnose(batch, reference, config, as_of_h=float(h), kb=camp.kb).reports.get(
                    t.batch_id
                )
                if rep and rep.leading and rep.leading.id == "H01":
                    lead_named = float(h)
                    break
        rows.append(
            {
                "batch_id": t.batch_id,
                "category": t.category,
                "severity": t.severity,
                "onset_h": t.onset_h,
                "any_alert": bool(mine),
                "flagged_after_onset": bool(after),
                "first_alert_h": first,
                "onset_to_alert_h": first - t.onset_h if after else np.nan,
                "contamination_leading_h": lead_named,
                "onset_to_named_h": lead_named - t.onset_h,
                "end_leading": table["leading_explanation"].get(t.batch_id, "(no alerts)")
                if len(table)
                else "(no alerts)",
                "end_status": table["contamination_evidence"].get(t.batch_id, "")
                if len(table)
                else "",
                "variables_flagged": ", ".join(sorted({a.variable for a in mine})),
            }
        )
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "batch_results.csv", index=False)

    normal = df[df["category"] == "normal"]
    summary = []
    for sev in SEVERITIES:
        c = df[df["severity"] == sev]
        summary.append(
            {
                "severity": sev,
                "contaminated batches flagged": rate(int(c["flagged_after_onset"].sum()), len(c)),
                "onset to first alert": median_text(c["onset_to_alert_h"]),
                "contamination named (any time)": rate(
                    int(c["contamination_leading_h"].notna().sum()), len(c)
                ),
                "onset to contamination named": median_text(c["onset_to_named_h"]),
            }
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(
        "SEMI-SYNTHETIC: IndPenSim background, SIMULATED contaminant effects (our "
        "assumptions). A software test, not real-world performance.\n"
    )
    print(
        "Unmodified held-out normal batches flagged:",
        rate(int(normal["any_alert"].sum()), len(normal)),
    )
    print("Normal batches with contamination as the leading explanation:",
          rate(int(normal["end_leading"].eq("Biological contamination").sum()), len(normal)))  # fmt: skip
    print(summary.to_string(index=False), "\n")
    print("Leading explanation at the end of contaminated batches:")
    print(
        df[df["category"] == "contamination"]
        .groupby(["severity", "end_leading"])
        .size()
        .to_string()
    )
    print("\nFirst variables flagged (contaminated batches):")
    first_vars = []
    for b in df.loc[df["category"] == "contamination", "batch_id"]:
        onset = truth.set_index("batch_id").loc[b, "onset_h"]
        a = sorted(
            (x for x in alerts if x.batch_id == b and x.start_h >= onset),
            key=lambda x: x.first_raised_h,
        )
        if a:
            first_vars.append(a[0].variable)
    print(pd.Series(first_vars).value_counts().to_string())
    return {"batches": df, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--with-shift", action="store_true", help="also run the optional shift detector"
    )
    args = parser.parse_args()
    config = DEFAULT_CONFIG
    if args.with_shift:
        config = replace(config, detectors=(*config.detectors, "shift"))
    print("Detectors:", ", ".join(config.detectors), "\n")
    out = OUT.with_name(OUT.name + ("_shift" if args.with_shift else ""))
    evaluate(out_dir=out, seed=args.seed, config=config)


if __name__ == "__main__":
    main()
