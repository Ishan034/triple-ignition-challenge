"""Multivariate benchmark: PCA with Hotelling's T² and SPE. COMPARISON ONLY.

The project uses simple, explainable per-variable detectors. This module answers the
reviewer's question "would a standard multivariate method do better?" on the same
data, under the same rules:

* Inputs are the same hourly robust z-scores (each variable's deviation from its
  reference median at that hour), for the signals measured every hour. Offline assays
  are left out because PCA needs a complete row at each hour.
* The PCA model is fitted on approved reference batches only. Each reference batch's
  z-scores come from a reference built without it, so calibration sees realistic
  out-of-sample deviations.
* T² measures unusual movement along the normal correlation structure; SPE (squared
  prediction error) measures a break in that structure. Limits are calibrated by
  5-fold cross-validation over reference batches, then tightened under the same
  batch-level false-alarm budget as the univariate detectors.
* An alert is any hour over either limit. The variable with the largest SPE
  contribution is reported, the usual first step in reading a PCA alarm.

    python -m contamination.multivariate indpensim
    python -m contamination.multivariate synthetic --seeds 11 31
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .monitor import DEFAULT_CONFIG, MonitorConfig, score
from .reference import build_reference
from .uncertainty import median_text, rate

N_FOLDS = 5


def hourly_z(scores: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """One row per (batch, hour) with every variable's z; incomplete rows dropped."""
    s = scores[scores["variable"].isin(variables)].copy()
    s["hour"] = s["elapsed_h"].round()
    z = s.pivot_table(index=["batch_id", "hour"], columns="variable", values="z", aggfunc="mean")
    return z.reindex(columns=variables).dropna()


def loo_z(measurements: pd.DataFrame, approved: list[str], config: MonitorConfig, variables):
    """Hourly z for each approved batch, scored against a reference built without it."""
    parts = []
    for b in approved:
        ref = build_reference(measurements, [x for x in approved if x != b], config.window_h)
        parts.append(
            hourly_z(score(measurements[measurements["batch_id"] == b], ref, config), variables)
        )
    return pd.concat(parts)


@dataclass
class PCAModel:
    mean: np.ndarray
    std: np.ndarray
    loadings: np.ndarray  # variables x components
    eigenvalues: np.ndarray

    @classmethod
    def fit(cls, z: np.ndarray, explained: float = 0.9) -> PCAModel:
        mean, std = z.mean(axis=0), z.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        x = (z - mean) / std
        _, sv, vt = np.linalg.svd(x, full_matrices=False)
        eig = sv**2 / (len(x) - 1)
        k = int(np.searchsorted(np.cumsum(eig) / eig.sum(), explained) + 1)
        return cls(mean, std, vt[:k].T, eig[:k])

    def statistics(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """T², SPE and per-variable SPE contributions."""
        x = (z - self.mean) / self.std
        t = x @ self.loadings
        resid = x - t @ self.loadings.T
        return (t**2 / self.eigenvalues).sum(axis=1), (resid**2).sum(axis=1), resid**2


def _batch_peaks(model: PCAModel, z: pd.DataFrame) -> pd.DataFrame:
    t2, spe, _ = model.statistics(z.to_numpy())
    return pd.DataFrame({"t2": t2, "spe": spe}, index=z.index).groupby(level="batch_id").max()


def calibrate(z_ref: pd.DataFrame, budget: float | None, margin: float = 1.1, seed: int = 0):
    """PCA on all reference rows; T² and SPE limits from cross-validated batch peaks."""
    batches = np.array(sorted(z_ref.index.get_level_values("batch_id").unique()))
    folds = np.array_split(np.random.default_rng(seed).permutation(batches), N_FOLDS)
    peaks = []
    for held in folds:
        inside = ~z_ref.index.get_level_values("batch_id").isin(held)
        model = PCAModel.fit(z_ref[inside].to_numpy())
        peaks.append(_batch_peaks(model, z_ref[~inside]))
    peaks = pd.concat(peaks)
    q, allowed = 1.0, (budget or 0.0) * len(batches)
    limits = {c: margin * peaks[c].quantile(q) for c in ("t2", "spe")}
    if budget is not None:
        for trial in np.round(np.arange(1.0, 0.795, -0.005), 3):
            lim = {c: margin * peaks[c].quantile(trial) for c in ("t2", "spe")}
            n = int(((peaks["t2"] > lim["t2"]) | (peaks["spe"] > lim["spe"])).sum())
            if n > allowed:
                break
            q, limits = float(trial), lim
    return PCAModel.fit(z_ref.to_numpy()), limits, q


def alert_hours(model: PCAModel, limits: dict, z: pd.DataFrame) -> pd.DataFrame:
    """Every hour over a limit, with the variable contributing most to SPE there."""
    t2, spe, contrib = model.statistics(z.to_numpy())
    over = np.flatnonzero((t2 > limits["t2"]) | (spe > limits["spe"]))
    return pd.DataFrame(
        {
            "batch_id": z.index.get_level_values("batch_id")[over],
            "hour": z.index.get_level_values("hour")[over].astype(float),
            "top_variable": z.columns[np.argmax(contrib[over], axis=1)] if len(over) else [],
        }
    )


def compare(m, reference_ids, monitored_ids, outcomes, variables, alerts, config):
    """Run the PCA benchmark and line it up with the univariate alerts on the same signals."""
    z_ref = loo_z(m, reference_ids, config, variables)
    model, limits, q = calibrate(z_ref, config.false_alarm_budget, config.calibration_margin)
    ref = build_reference(m, reference_ids, config.window_h)
    z_test = hourly_z(score(m[m["batch_id"].isin(monitored_ids)], ref, config), variables)
    mv = alert_hours(model, limits, z_test)
    rows = []
    for o in outcomes.itertuples(index=False):
        onset = o.onset_h if pd.notna(o.onset_h) else -np.inf
        uni = [a for a in alerts if a.batch_id == o.batch_id and a.variable in variables]
        uni_after = [a.first_raised_h for a in uni if a.start_h >= onset]
        pca = mv[mv["batch_id"] == o.batch_id]
        pca_after = pca[pca["hour"] >= onset]
        rows.append(
            {
                "batch_id": o.batch_id,
                "category": o.category,
                "scenario": o.scenario,
                "onset_h": o.onset_h,
                "univariate_any": bool(uni),
                "univariate_first_h": min(uni_after) if uni_after else np.nan,
                "pca_any": bool(len(pca)),
                "pca_first_h": float(pca_after["hour"].min()) if len(pca_after) else np.nan,
                "pca_top_variable": pca_after["top_variable"].iloc[0] if len(pca_after) else "",
            }
        )
    info = {"quantile": q, "limits": limits, "components": model.loadings.shape[1]}
    return pd.DataFrame(rows), info


def summarise(df: pd.DataFrame, label: str) -> pd.DataFrame:
    normal, problems = df[df["category"] == "normal"], df[df["category"] != "normal"]
    out = []
    for name, prefix in (("univariate (same signals)", "univariate"), ("PCA T²/SPE", "pca")):
        caught = problems[f"{prefix}_first_h"].notna()
        delay = (
            problems.loc[caught, f"{prefix}_first_h"] - problems.loc[caught, "onset_h"]
        ).to_numpy(float)
        out.append(
            {
                "dataset": label,
                "method": name,
                "normal batches flagged": rate(int(normal[f"{prefix}_any"].sum()), len(normal)),
                "problem batches caught": rate(int(caught.sum()), len(problems)),
                "onset to alert": median_text(delay),
            }
        )
    return pd.DataFrame(out)


def run_indpensim(campaign_dir: Path = Path("data/indpensim")):
    from .monitor import monitor
    from .pipeline import fit, load_campaign

    camp = load_campaign(campaign_dir)
    reference = fit(camp)
    m = camp.measurements
    alerts = monitor(m[m["batch_id"].isin(camp.monitored_ids)], reference).alerts
    variables = ["ph", "temperature", "do2_mg_l", "our", "cer", "co2_offgas", "substrate"]
    variables += [v for v in ("base_flow", "coolant_flow") if v in set(m["variable"])]
    outcomes = camp.outcomes[camp.outcomes["batch_id"].isin(camp.monitored_ids)]
    df, info = compare(
        m, camp.reference_ids, camp.monitored_ids, outcomes, variables, alerts, DEFAULT_CONFIG
    )
    return df, info


def run_synthetic(seeds) -> tuple[pd.DataFrame, list[dict]]:
    from .diagnosis import load_knowledge
    from .ingest import validate_measurements
    from .monitor import fit_reference, monitor
    from .synthetic import generate_dataset

    variables = ["ph", "temperature", "dissolved_oxygen"]
    parts, infos = [], []
    for seed in seeds:
        ds = generate_dataset(seed=seed, n_reference=30)
        m = validate_measurements(ds.measurements).measurements
        ordered = ds.batches.sort_values("start_time")["batch_id"].tolist()
        ref_ids, test_ids = ordered[:30], ordered[30:]
        reference = fit_reference(m, ref_ids, linked_signals=list(load_knowledge().linked_signals))
        alerts = monitor(m[m["batch_id"].isin(test_ids)], reference).alerts
        outcomes = ds.outcomes[ds.outcomes["batch_id"].isin(test_ids)]
        df, info = compare(m, ref_ids, test_ids, outcomes, variables, alerts, DEFAULT_CONFIG)
        df["seed"] = seed
        parts.append(df)
        infos.append(info)
    return pd.concat(parts, ignore_index=True), infos


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", choices=["indpensim", "synthetic"])
    parser.add_argument("--seeds", type=int, nargs=2, default=(11, 31))
    parser.add_argument("--out", type=Path, default=Path("outputs/multivariate"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    if args.dataset == "indpensim":
        df, info = run_indpensim()
        print(
            "PCA model:", info["components"], "components; calibration quantile", info["quantile"]
        )
        table = summarise(df, "IndPenSim (hourly signals)")
        faults = df[df["category"] != "normal"]
        print(
            faults[
                [
                    "batch_id",
                    "scenario",
                    "onset_h",
                    "univariate_first_h",
                    "pca_first_h",
                    "pca_top_variable",
                ]
            ].to_string(index=False)
        )
    else:
        df, infos = run_synthetic(range(*args.seeds))
        table = pd.concat(
            [
                summarise(
                    df[df["category"].isin(["normal", "contamination"])], "synthetic: contamination"
                ),
                summarise(df[df["category"] != "contamination"], "synthetic: other anomalies"),
            ]
        )
        print("PCA components per seed:", [i["components"] for i in infos])
    df.to_csv(args.out / f"{args.dataset}_batches.csv", index=False)
    table.to_csv(args.out / f"{args.dataset}_summary.csv", index=False)
    print("\nCOMPARISON ONLY: the project's detectors stay the simple per-variable ones.\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
