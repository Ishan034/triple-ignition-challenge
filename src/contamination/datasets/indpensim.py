"""IndPenSim: 100 simulated batches of a 100,000 L fed-batch penicillin process.

Source: Goldrick et al. (2015) J. Biotechnol. 193:70-82, data on Mendeley
(doi:10.17632/pdnjz7zz5x, CC BY 4.0). Batches 1-30 are recipe driven, 31-60 operator
controlled, 61-90 run by advanced process control (Raman), 91-100 contain process
faults. The faults are equipment/actuator faults (aeration, substrate feed, base
flow, coolant flow), NOT contamination, so this dataset tests false alarms on
realistic variation, early detection of real process deviations, and whether the
diagnosis avoids calling an equipment fault "contamination".

    python -m contamination.datasets.indpensim prepare    # download + convert
    python -m contamination.datasets.indpensim evaluate   # run and score the model

Conversion choices (documented, not hidden):

* Process outputs are monitored (pH, temperature, DO, OUR, CER, off-gas CO2,
  substrate, offline biomass and penicillin), plus the outputs of the pH and
  temperature controllers (base flow, cooling-water flow), which a plant historian
  records. The recipe-driven inputs that the aeration and substrate-feed faults act
  on (aeration rate, sugar feed rate) are excluded: watching the input that was
  changed would make those detections trivial. The base-flow and coolant faults act
  on the controller outputs themselves, so their detection is partly by construction;
  what matters for them is the diagnosis (equipment fault, not contamination).
* Online signals (logged every 0.2 h) are kept at whole hours, as a plant historian
  would. Offline assays (every 12 h) are given a 4 h result delay, IndPenSim's
  default offline-assay delay, so they only count once available.
* Temperature is converted from K to degC. Batches get sequential synthetic start
  dates; IndPenSim records no lots or equipment, so source tracing does not apply.
"""

from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..diagnosis import PROFILES_DIR
from ..schema import MEASUREMENT_COLUMNS, VARIABLES

ZIP_URL = (
    "https://data.mendeley.com/public-files/datasets/pdnjz7zz5x/files/"
    "e7007dff-4061-494f-a9f5-16c20f457d1d/file_downloaded"
)
RAW_DIR = Path("data/external/indpensim")
CAMPAIGN_DIR = RAW_DIR / "campaign"
N_PROCESS_COLUMNS = 39  # the rest of each row is 2,200 Raman wavenumbers

ONLINE = {
    "pH(pH:pH)": "ph",
    "Temperature(T:K)": "temperature",
    "Dissolved oxygen concentration(DO2:mg/L)": "do2_mg_l",
    "Oxygen Uptake Rate(OUR:(g min^{-1}))": "our",
    "Carbon evolution rate(CER:g/h)": "cer",
    "carbon dioxide percent in off-gas(CO2outgas:%)": "co2_offgas",
    "Substrate concentration(S:g/L)": "substrate",
    # Controller outputs of the pH and temperature loops.
    "Base flow rate(Fb:L/h)": "base_flow",
    "Heating/cooling water flow rate(Fc:L/h)": "coolant_flow",
}
OFFLINE = {
    "Offline Biomass concentratio(X_offline:X(g L^{-1}))": "biomass_offline",
    "Offline Penicillin concentration(P_offline:P(g L^{-1}))": "penicillin_offline",
}
OFFLINE_DELAY_H = 4.0
FAULT_COL = "Fault reference(Fault_ref:Fault ref)"
CONTROL_COL = "0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)"

# IndPenSim's built-in fault types, recognised by the timing of their first window
# (start h, end h), as coded in the simulator (PenSimPy peni_env_setup.py).
FAULT_SIGNATURES = {
    (20.0, 24.0): "aeration rate fault",
    (100.0, 104.0): "vessel back-pressure fault",
    (20.0, 30.0): "substrate feed rate fault",
    (80.0, 84.0): "base flow-rate fault",
    (70.0, 90.0): "coolant flow-rate fault",
}


def control_strategy(batch_no: int) -> str:
    if batch_no <= 30:
        return "recipe"
    if batch_no <= 60:
        return "operator"
    if batch_no <= 90:
        return "apc"
    return "fault"


def download(raw_dir: Path = RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "100_Batches_IndPenSim.zip"
    if not target.exists():
        print(f"downloading {ZIP_URL} (~500 MB)")
        with urllib.request.urlopen(ZIP_URL) as r, open(target, "wb") as f:
            shutil.copyfileobj(r, f)
    return target


def extract_process(zip_path: Path, raw_dir: Path = RAW_DIR) -> Path:
    """Stream the 2.5 GB CSV and keep only the 39 process columns (~22 MB)."""
    out_path = raw_dir / "indpensim_process.csv"
    if out_path.exists():
        return out_path
    z = zipfile.ZipFile(zip_path)
    with z.open("Mendeley_data/100_Batches_IndPenSim_V3.csv") as f, open(out_path, "w") as out:
        out.writelines(
            ",".join(raw.rstrip("\n").split(",", N_PROCESS_COLUMNS)[:N_PROCESS_COLUMNS]) + "\n"
            for raw in io.TextIOWrapper(f, encoding="utf-8", errors="replace")
        )
    with z.open("Mendeley_data/100_Batches_IndPenSim_Statistics.csv") as f:
        (raw_dir / "indpensim_statistics.csv").write_bytes(f.read())
    return out_path


def fault_windows(
    times: np.ndarray, flags: np.ndarray, step_h: float = 0.2
) -> list[tuple[float, float]]:
    t = times[flags > 0]
    if not len(t):
        return []
    breaks = np.where(np.diff(t) > 1.5 * step_h)[0]
    starts, ends = np.r_[t[0], t[breaks + 1]], np.r_[t[breaks], t[-1]]
    return [(round(float(a), 1), round(float(b), 1)) for a, b in zip(starts, ends)]


def convert(process_csv: Path, out_dir: Path = CAMPAIGN_DIR) -> dict[str, pd.DataFrame]:
    d = pd.read_csv(process_csv)
    d.columns = [c.strip() for c in d.columns]
    missing = (set(ONLINE) | set(OFFLINE) | {"Time (h)", FAULT_COL}) - set(d.columns)
    if missing:
        raise ValueError(f"IndPenSim columns not found: {sorted(missing)}")
    d["batch_no"] = (d["Time (h)"].diff().fillna(-1) < 0).cumsum()
    if d["batch_no"].nunique() != 100:
        raise ValueError(f"expected 100 batches, found {d['batch_no'].nunique()}")

    t0 = pd.Timestamp("2026-01-05 08:00")
    rows, batches, outcomes = [], [], []
    for no, b in d.groupby("batch_no"):
        batch_id = f"P{no:03d}"
        start = t0 + pd.Timedelta(days=12 * (no - 1))
        times = b["Time (h)"].to_numpy()
        hourly = b[np.isclose(times, np.round(times))]
        for col, var in ONLINE.items():
            vals = hourly[col].to_numpy(float)
            if var == "temperature":
                vals = vals - 273.15
            for h, v in zip(hourly["Time (h)"].to_numpy(), vals):
                ts = start + pd.Timedelta(hours=float(h))
                rows.append((batch_id, ts, ts, var, v, "online_sensor"))
        for col, var in OFFLINE.items():
            off = b[b[col].notna()]
            for h, v in zip(off["Time (h)"].to_numpy(), off[col].to_numpy(float)):
                ts = start + pd.Timedelta(hours=float(h))
                rows.append(
                    (
                        batch_id,
                        ts,
                        ts + pd.Timedelta(hours=OFFLINE_DELAY_H),
                        var,
                        v,
                        "offline_assay",
                    )
                )

        windows = fault_windows(times, b[FAULT_COL].to_numpy())
        strategy = control_strategy(no)
        fault_type = (
            FAULT_SIGNATURES.get(windows[0], "unrecognised fault pattern") if windows else ""
        )
        batches.append(
            {
                "batch_id": batch_id,
                "start_time": start,
                "bioreactor_id": "IndPenSim 100 kL",
                "medium_lot": "not recorded",
                "inoculum_lot": "not recorded",
                "control_strategy": strategy,
                "batch_length_h": float(times.max()),
            }
        )
        outcomes.append(
            {
                "batch_id": batch_id,
                "category": "non_contamination_anomaly" if windows else "normal",
                "scenario": fault_type.replace(" ", "_") if windows else f"normal_{strategy}",
                "outcome_status": "confirmed",
                "onset_h": windows[0][0] if windows else np.nan,
                "reference_event_h": np.nan,
                "fault_windows": "; ".join(f"{a:g}-{b:g} h" for a, b in windows),
                "simulated_source": "",
                "reason": f"IndPenSim {fault_type} ({len(windows)} window(s))"
                if windows
                else f"IndPenSim normal batch ({strategy} control)",
                "data_origin": "IndPenSim simulation (Goldrick et al. 2015), CC BY 4.0",
            }
        )

    m = pd.DataFrame(
        rows, columns=["batch_id", "timestamp", "available_at", "variable", "value", "source"]
    )
    m["stage"] = np.nan
    m["unit"] = m["variable"].map({k: v.unit for k, v in VARIABLES.items()})
    m["quality_flag"] = np.where(np.isfinite(m["value"]), "valid", "missing")
    m["observation"] = None
    m = m[MEASUREMENT_COLUMNS]

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "measurements": m,
        "batches": pd.DataFrame(batches),
        "outcomes": pd.DataFrame(outcomes),
    }
    for name, df in result.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    shutil.copy(PROFILES_DIR / "indpensim.yaml", out_dir / "profile.yaml")
    return result


def reference_batches() -> list[str]:
    """Approved reference: the first 20 batches of each control strategy."""
    return [f"P{n:03d}" for n in list(range(1, 21)) + list(range(31, 51)) + list(range(61, 81))]


def prepare(raw_dir: Path = RAW_DIR) -> Path:
    convert(extract_process(download(raw_dir), raw_dir), raw_dir / "campaign")
    return raw_dir / "campaign"


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
        from .evaluate_external import evaluate_indpensim

        evaluate_indpensim(args.raw_dir / "campaign")


if __name__ == "__main__":
    main()
