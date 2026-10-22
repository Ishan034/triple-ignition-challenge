"""Import a plant's own data exports into a campaign folder, then check readiness.

Plant data never arrives in our format. A small ``mapping.yaml`` written next to the
exports says which file and column holds what; this module turns those exports into
the campaign tables the pipeline reads (``measurements.csv``, ``batches.csv``,
``outcomes.csv``, ``investigations.csv``, ``profile.yaml``, ``variables.yaml``).

    python -m contamination.importer convert path/to/mapping.yaml --out data/maverickx/campaign
    python -m contamination.importer readiness data/maverickx/campaign

Conversion follows the ingestion rules: values are copied as they are (never filled,
rescaled or dropped), unknown columns are reported rather than guessed, and every
measurement keeps when it was taken and when its result became available.

``readiness`` reports what the data supports before any model is run: which process
roles are measured, how many approved normal batches and confirmed contamination
cases exist, whether result times and genealogy are recorded, and what is missing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .schema import MEASUREMENT_COLUMNS, OUTCOME_CATEGORIES, VARIABLES, VariableSpec

MIN_NORMAL_BATCHES = 30  # reference size used in our evaluations
MIN_CONTAMINATION_CASES = 10
GENEALOGY_COLUMNS = ("bioreactor_id", "medium_lot", "inoculum_lot")
NOT_RECORDED = "not recorded"


# --------------------------------------------------------------------------- variables


def register_variables(spec: dict | None) -> list[str]:
    """Add process-specific variables (from a mapping or ``variables.yaml``) to the registry.

    Each entry needs ``unit``, ``min_scale`` (noise floor used when the reference spread
    is ~0) and ``plausible_range``. Built-in variables cannot be redefined.
    """
    added = []
    for name, v in (spec or {}).items():
        new = VariableSpec(
            str(v["unit"]),
            float(v["min_scale"]),
            (float(v["plausible_range"][0]), float(v["plausible_range"][1])),
            str(v.get("description", name)),
        )
        if name in VARIABLES and VARIABLES[name] != new:
            raise ValueError(f"variable {name!r} is already defined differently")
        VARIABLES[name] = new
        added.append(name)
    return added


def load_variables(campaign_dir: str | Path) -> list[str]:
    path = Path(campaign_dir) / "variables.yaml"
    return register_variables(yaml.safe_load(path.read_text())) if path.exists() else []


# --------------------------------------------------------------------------- conversion


def _read(base: Path, spec: dict) -> pd.DataFrame:
    path = base / spec["file"]
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=spec.get("sheet", 0))
    return pd.read_csv(path, sep=spec.get("sep", ","))


def _require(df: pd.DataFrame, cols, where: str) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise ValueError(f"{where}: columns not found {missing}; found {list(df.columns)}")


def _times(series: pd.Series, spec: dict) -> pd.Series:
    t = pd.to_datetime(series, format=spec.get("timestamp_format"))
    tz = spec.get("timezone")
    if tz:
        t = t.dt.tz_localize(tz) if t.dt.tz is None else t.dt.tz_convert(tz)
        t = t.dt.tz_localize(None)
    return t


def _measurement_rows(
    base: Path, spec: dict, starts: pd.Series, unmapped: list[str]
) -> pd.DataFrame:
    df = _read(base, spec)
    where = spec["file"]
    _require(df, [spec["batch_id"]], where)
    df = df.rename(columns={spec["batch_id"]: "batch_id"})
    df["batch_id"] = df["batch_id"].astype(str)

    # Time: either a timestamp column, or elapsed hours from the batch start.
    if "timestamp" in spec:
        _require(df, [spec["timestamp"]], where)
        df["timestamp"] = _times(df[spec["timestamp"]], spec)
    elif "elapsed_hours" in spec:
        _require(df, [spec["elapsed_hours"]], where)
        start = df["batch_id"].map(starts)
        if start.isna().any():
            raise ValueError(f"{where}: elapsed hours need a start_time for every batch")
        df["timestamp"] = start + pd.to_timedelta(df[spec["elapsed_hours"]], unit="h")
    else:
        raise ValueError(f"{where}: give either 'timestamp' or 'elapsed_hours'")

    if spec.get("layout", "wide") == "wide":
        columns = spec["columns"]
        _require(df, list(columns), where)
        known = {spec["batch_id"], spec.get("timestamp"), spec.get("elapsed_hours")}
        unmapped += [
            f"{where}: {c}"
            for c in df.columns
            if c not in columns and c not in known and c not in ("batch_id", "timestamp")
        ]
        long = df.melt(
            id_vars=["batch_id", "timestamp"],
            value_vars=list(columns),
            var_name="column",
            value_name="value",
        )
        long["variable"] = long["column"].map(columns)
        long["unit"] = long["variable"].map(spec.get("units", {}))
    else:
        _require(df, [spec["variable"], spec["value"]], where)
        vmap = spec.get("variable_map", {})
        long = df.rename(columns={spec["value"]: "value"})
        raw_names = long[spec["variable"]].astype(str)
        unmapped += sorted(f"{where}: {v}" for v in set(raw_names) - set(vmap))
        long = long[raw_names.isin(vmap)].copy()
        long["variable"] = long[spec["variable"]].astype(str).map(vmap)
        long["unit"] = long[spec["unit"]] if spec.get("unit") in df.columns else np.nan
        if "available_at" in spec:
            _require(df, [spec["available_at"]], where)
            long["available_at"] = _times(long[spec["available_at"]], spec)

    if "available_at" not in long:
        delay = float(spec.get("result_delay_h", 0.0))
        long["available_at"] = long["timestamp"] + pd.to_timedelta(delay, unit="h")
    # Unit defaults to the variable's registered unit only when the export has none.
    default_unit = long["variable"].map({k: v.unit for k, v in VARIABLES.items()})
    long["unit"] = long["unit"].where(long["unit"].notna(), default_unit)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["source"] = spec.get("source", "plant_export")
    long["stage"] = np.nan
    long["quality_flag"] = np.where(long["value"].notna(), "valid", "missing")
    long["observation"] = None
    return long[MEASUREMENT_COLUMNS]


def _observation_rows(base: Path, spec: dict) -> pd.DataFrame:
    df = _read(base, spec)
    _require(df, [spec["batch_id"], spec["timestamp"], spec["text"]], spec["file"])
    t = _times(df[spec["timestamp"]], spec)
    return pd.DataFrame(
        {
            "batch_id": df[spec["batch_id"]].astype(str),
            "timestamp": t,
            "available_at": t,
            "stage": np.nan,
            "variable": spec.get("variable", "ph"),
            "value": np.nan,
            "unit": None,
            "source": "operator_note",
            "quality_flag": "missing",
            "observation": df[spec["text"]].astype(str),
        }
    )[MEASUREMENT_COLUMNS]


def _table(base: Path, spec: dict, required: tuple[str, ...]) -> pd.DataFrame:
    df = _read(base, spec)
    cols = spec["columns"]
    _require(df, list(cols.values()), spec["file"])
    out = df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
    for c in required:
        if c not in out:
            out[c] = NOT_RECORDED
    out["batch_id"] = out["batch_id"].astype(str)
    return out


@dataclass
class ImportResult:
    out_dir: Path
    tables: dict[str, pd.DataFrame]
    unmapped: list[str]


def convert(mapping_path: str | Path, out_dir: str | Path) -> ImportResult:
    """Convert plant exports described by ``mapping.yaml`` into a campaign folder."""
    mapping_path = Path(mapping_path)
    base, m = mapping_path.parent, yaml.safe_load(mapping_path.read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    register_variables(m.get("variables"))

    batches = _table(base, m["batches"], ("batch_id", "start_time", *GENEALOGY_COLUMNS))
    batches["start_time"] = _times(batches["start_time"], m["batches"])
    for c in GENEALOGY_COLUMNS:
        batches[c] = batches[c].fillna(NOT_RECORDED).astype(str)
    starts = batches.set_index("batch_id")["start_time"]

    unmapped: list[str] = []
    parts = [_measurement_rows(base, s, starts, unmapped) for s in m["measurements"]]
    parts += [_observation_rows(base, s) for s in m.get("observations", [])]
    measurements = pd.concat(parts, ignore_index=True)
    unknown_batches = sorted(set(measurements["batch_id"]) - set(batches["batch_id"]))
    if unknown_batches:
        raise ValueError(
            f"measurements for batches missing from the batch table: {unknown_batches[:10]}"
        )

    tables = {"measurements": measurements, "batches": batches}
    if "outcomes" in m:
        o = _table(base, m["outcomes"], ("batch_id", "category"))
        cmap = m["outcomes"].get("category_map")
        if cmap:
            o["category"] = o["category"].astype(str).map(cmap)
        bad = sorted(set(o["category"].dropna()) - set(OUTCOME_CATEGORIES))
        if bad or o["category"].isna().any():
            raise ValueError(f"outcome categories must map to {OUTCOME_CATEGORIES}; got {bad}")
        o["scenario"] = o.get("scenario", o["category"])
        o["outcome_status"] = o.get("outcome_status", "reviewed")
        for c in ("onset_h", "reference_event_h"):
            o[c] = pd.to_numeric(o[c], errors="coerce") if c in o else np.nan
        o["reason"] = o.get("reason", "imported from plant records")
        tables["outcomes"] = o
    if "investigations" in m:
        tables["investigations"] = _table(
            base, m["investigations"], ("batch_id", "check_id", "result", "recorded_h", "note")
        )

    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "variables.yaml").write_text(yaml.safe_dump(m.get("variables") or {}, sort_keys=False))
    if "roles" in m:
        profile = {
            "name": m.get("process_name", "imported process"),
            "roles": m["roles"],
            "parameters": m.get("parameters", {}),
        }
        (out / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    return ImportResult(out, tables, unmapped)


# --------------------------------------------------------------------------- example


EXAMPLE_MAPPING = """\
# Example mapping: how a plant's exports map onto the campaign tables.
# Copy this file next to your own exports and change file and column names to match.
process_name: Example enzyme fermentation (synthetic demo data)

# Variables the built-in list does not have. Each needs a unit, a noise floor
# (min_scale: the smallest spread that counts as real) and a plausible range.
variables: {}

measurements:
  - file: historian_export.csv     # online signals, one column per signal ("wide")
    layout: wide
    batch_id: Batch
    timestamp: Time
    source: online_sensor
    columns:                        # export column -> variable name
      pH: ph
      Temp (C): temperature
      DO (%sat): dissolved_oxygen
  - file: lab_results.csv          # offline results, one row per result ("long")
    layout: long
    batch_id: Batch
    timestamp: Sampled              # when the sample was taken
    available_at: Reported          # when the result came back
    variable: Test
    value: Result
    unit: Units
    source: offline_assay
    variable_map:                   # test name in the export -> variable name
      OD600: biomass_od600
      Amylase activity: enzyme_activity

observations:                       # timestamped operator notes (smell, colour, foam)
  - file: operator_notes.csv
    batch_id: Batch
    timestamp: Time
    text: Note

batches:
  file: batch_records.csv
  columns:                          # campaign column -> export column
    batch_id: Batch
    start_time: Inoculated
    bioreactor_id: Vessel
    medium_lot: Medium lot
    inoculum_lot: Seed lot

outcomes:                           # kept apart from measurements; used only to evaluate
  file: batch_outcomes.csv
  columns:
    batch_id: Batch
    category: Outcome
    onset_h: Problem started (h)     # elapsed hours, if known (e.g. from a later investigation)
    reference_event_h: First noticed (h)  # when today's checks first noticed it
    reason: Comment
  category_map:
    OK: normal
    Contaminated: contamination
    Other problem: non_contamination_anomaly

# Which variable plays each role in the diagnosis rules. Controller outputs
# (ph_control, temperature_control) are optional: "not_measured" if not recorded.
roles:
  biomass: biomass_od600
  oxygen: dissolved_oxygen
  ph: ph
  product: enzyme_activity
  temperature: temperature
  ph_control: not_measured
  temperature_control: not_measured
"""


def write_example_exports(campaign_dir: str | Path, out_dir: str | Path, batch_ids=None) -> Path:
    """Write plant-style exports (and a matching ``mapping.yaml``) from a campaign.

    Used to build the data template for MaverickX and to test the importer round trip.
    """
    from .pipeline import load_campaign

    camp = load_campaign(campaign_dir, reference_ids=[])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = camp.measurements
    keep = set(batch_ids) if batch_ids is not None else set(camp.batches["batch_id"])
    m = m[m["batch_id"].isin(keep)]
    online = {"ph": "pH", "temperature": "Temp (C)", "dissolved_oxygen": "DO (%sat)"}
    lab = {"biomass_od600": "OD600", "enzyme_activity": "Amylase activity"}
    fmt = "%Y-%m-%d %H:%M"

    on = m[m["variable"].isin(online) & m["observation"].isna()]
    wide = (
        on.groupby(["batch_id", "timestamp", "variable"])["value"].first().unstack().reset_index()
    )
    wide = wide.rename(columns={"batch_id": "Batch", **online})
    wide["Time"] = wide.pop("timestamp").dt.strftime(fmt)
    wide[["Batch", "Time", *online.values()]].to_csv(out / "historian_export.csv", index=False)

    off = m[m["variable"].isin(lab)]
    pd.DataFrame(
        {
            "Batch": off["batch_id"],
            "Sampled": off["timestamp"].dt.strftime(fmt),
            "Reported": off["available_at"].dt.strftime(fmt),
            "Test": off["variable"].map(lab),
            "Result": off["value"],
            "Units": off["unit"],
        }
    ).to_csv(out / "lab_results.csv", index=False)

    notes = m[m["observation"].notna()]
    pd.DataFrame(
        {
            "Batch": notes["batch_id"],
            "Time": notes["timestamp"].dt.strftime(fmt),
            "Note": notes["observation"],
        }
    ).to_csv(out / "operator_notes.csv", index=False)

    b = camp.batches[camp.batches["batch_id"].isin(keep)]
    pd.DataFrame(
        {
            "Batch": b["batch_id"],
            "Inoculated": b["start_time"].dt.strftime(fmt),
            "Vessel": b["bioreactor_id"],
            "Medium lot": b["medium_lot"],
            "Seed lot": b["inoculum_lot"],
        }
    ).to_csv(out / "batch_records.csv", index=False)

    if camp.outcomes is not None:
        o = camp.outcomes[camp.outcomes["batch_id"].isin(keep)]
        names = {
            "normal": "OK",
            "contamination": "Contaminated",
            "non_contamination_anomaly": "Other problem",
        }
        pd.DataFrame(
            {
                "Batch": o["batch_id"],
                "Outcome": o["category"].map(names),
                "Problem started (h)": o["onset_h"],
                "First noticed (h)": o["reference_event_h"],
                "Comment": o["reason"],
            }
        ).to_csv(out / "batch_outcomes.csv", index=False)
    (out / "mapping.yaml").write_text(EXAMPLE_MAPPING)
    return out


# --------------------------------------------------------------------------- readiness


def readiness(campaign_dir: str | Path) -> pd.DataFrame:
    """What this campaign supports, and what is missing, before the model is run."""
    from .diagnosis import NOT_MEASURED, OPTIONAL_ROLES, ROLES
    from .pipeline import load_campaign

    d = Path(campaign_dir)
    camp = load_campaign(d, reference_ids=[])
    m, batches, kb = camp.measurements, camp.batches, camp.kb
    rows: list[tuple[str, str, str, str]] = []

    def add(area, check, status, detail):
        rows.append((area, check, status, detail))

    n_batches = batches["batch_id"].nunique()
    add("Batches", "Batches in the batch table", "ok" if n_batches else "missing", f"{n_batches}")
    for role in (*ROLES, *OPTIONAL_ROLES):
        var = kb.roles.get(role, NOT_MEASURED)
        if var == NOT_MEASURED:
            status = "warn" if role in OPTIONAL_ROLES else "missing"
            add("Roles", f"{role}", status, "not mapped to a measured variable")
            continue
        present = m[(m["variable"] == var) & m["value"].notna()]
        cover = present["batch_id"].nunique()
        status = "ok" if cover == n_batches else ("warn" if cover else "missing")
        add("Roles", f"{role}", status, f"{var}: measured in {cover}/{n_batches} batches")

    for var, g in m[m["value"].notna()].groupby("variable"):
        step = g.sort_values("elapsed_h").groupby("batch_id")["elapsed_h"].diff().median()
        delay = (g["available_h"] - g["elapsed_h"]).median()
        add(
            "Signals",
            var,
            "ok" if var in VARIABLES else "warn",
            f"every {step:.3g} h, result delay {delay:.3g} h, {g['batch_id'].nunique()} batches",
        )
    issues = camp.ingest.summary()
    add(
        "Quality",
        "Readings flagged at ingestion",
        "ok" if issues.empty else "warn",
        ", ".join(f"{k}: {v}" for k, v in issues.items()) or "none",
    )

    if camp.outcomes is None:
        add("Outcomes", "Outcomes table", "missing", "no outcomes.csv: nothing to test against")
    else:
        o = camp.outcomes
        normal = int((o["category"] == "normal").sum())
        cont = o[o["category"] == "contamination"]
        add(
            "Outcomes",
            "Approved normal batches",
            "ok" if normal >= MIN_NORMAL_BATCHES else "warn",
            f"{normal} (want >= {MIN_NORMAL_BATCHES} for a reliable reference)",
        )
        add(
            "Outcomes",
            "Confirmed contamination cases",
            "ok" if len(cont) >= MIN_CONTAMINATION_CASES else ("warn" if len(cont) else "missing"),
            f"{len(cont)} (want >= {MIN_CONTAMINATION_CASES})",
        )
        timed = (
            int((cont["onset_h"].notna() | cont["reference_event_h"].notna()).sum())
            if len(cont)
            else 0
        )
        add(
            "Outcomes",
            "Contamination cases with an onset or first-noticed time",
            "ok" if len(cont) and timed == len(cont) else "warn",
            f"{timed}/{len(cont)}: needed to measure how much earlier the model warns",
        )
    for c in GENEALOGY_COLUMNS:
        filled = int((batches[c].astype(str) != NOT_RECORDED).sum())
        add(
            "Genealogy",
            c,
            "ok" if filled == n_batches else ("warn" if filled else "missing"),
            f"recorded for {filled}/{n_batches} batches",
        )
    add(
        "Investigations",
        "Recorded check results",
        "ok" if len(camp.investigations) else "warn",
        f"{len(camp.investigations)} rows",
    )
    return pd.DataFrame(rows, columns=["area", "check", "status", "detail"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("convert", help="convert plant exports using a mapping.yaml")
    c.add_argument("mapping", type=Path)
    c.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("readiness", help="report what a campaign folder supports")
    r.add_argument("campaign", type=Path)
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 90)
    if args.command == "convert":
        res = convert(args.mapping, args.out)
        print(f"Campaign written to {res.out_dir}:")
        for name, df in res.tables.items():
            print(f"  {name}.csv: {len(df)} rows")
        if res.unmapped:
            print("Columns/names in the exports that the mapping does not use (not imported):")
            print("  " + "\n  ".join(res.unmapped))
    else:
        print(readiness(args.campaign).to_string(index=False))


if __name__ == "__main__":
    main()
