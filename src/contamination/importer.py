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
    return convert_mapping(yaml.safe_load(mapping_path.read_text()), mapping_path.parent, out_dir)


def convert_mapping(m: dict, base: str | Path, out_dir: str | Path) -> ImportResult:
    """Convert exports in ``base`` using a mapping given as a dict (same keys as the YAML).

    The dashboard's column mapper builds this dict from the user's choices."""
    base = Path(base)
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
    if "use_profile" in m:  # follow a packaged profile (e.g. maverickx)
        (out / "profile.yaml").write_text(yaml.safe_dump({"use": m["use_profile"]}))
    elif "roles" in m:
        profile = {
            "name": m.get("process_name", "imported process"),
            "roles": m["roles"],
            "parameters": m.get("parameters", {}),
        }
        if m.get("linked_signals"):
            profile["linked_signals"] = m["linked_signals"]
        (out / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    return ImportResult(out, tables, unmapped)


# --------------------------------------------------------------------------- suggestions
# Used by the dashboard's column mapper to pre-fill its choices. Suggestions are only
# defaults shown to the user; nothing is imported until they confirm the mapping.

VARIABLE_HINTS: dict[str, tuple[str, ...]] = {  # variable -> name fragments (lower case)
    "ph_offline": ("ph offline", "offline ph", "ph (lab)", "lab ph"),
    "ph": ("ph",),
    "temperature": ("temp", "temperature"),
    "dissolved_oxygen": ("do (%", "do%", "dissolved oxygen", "do2 %", "po2", "do sat", "dot"),
    "do2_mg_l": ("do (mg", "do mg", "do2 mg", "dissolved oxygen (mg"),
    "biomass_od600": ("od600", "od 600", "optical density", "od"),
    "enzyme_activity": ("activity", "enzyme", "amylase", "titre", "titer"),
    "our": ("our", "oxygen uptake"),
    "cer": ("cer", "co2 evolution", "carbon evolution"),
    "co2_offgas": ("co2", "off-gas co2", "offgas co2", "exhaust co2"),
    "substrate": ("glucose", "substrate", "sugar"),
    "base_flow": ("base", "naoh", "ammonia", "nh4oh", "alkali"),
    "coolant_flow": ("cooling", "coolant", "jacket flow", "cooling water"),
    "agitation": ("agitation", "stirrer", "rpm", "impeller"),
}
COLUMN_HINTS: dict[str, tuple[str, ...]] = {  # field -> name fragments
    "batch_id": ("batch", "lot no", "run", "batch id", "batch_id"),
    "timestamp": ("time", "date", "timestamp", "sampled", "datetime"),
    "elapsed_hours": ("elapsed", "hours", "process time", "age", "eft", "(h)"),
    "available_at": ("reported", "result time", "available", "approved", "released"),
    "variable": ("test", "parameter", "analyte", "variable", "assay"),
    "value": ("result", "value", "reading"),
    "unit": ("unit", "units", "uom"),
    "text": ("note", "comment", "observation", "remark"),
    "start_time": ("inocul", "start", "began"),
    "bioreactor_id": ("vessel", "reactor", "fermenter", "fermentor", "tank"),
    "medium_lot": ("medium", "media"),
    "inoculum_lot": ("seed", "inoculum", "starter"),
    "category": ("outcome", "result", "status", "verdict", "disposition"),
    "onset_h": ("started", "onset"),
    "reference_event_h": ("noticed", "detected", "first seen"),
    "reason": ("comment", "reason", "note"),
}


def _norm(name: str) -> str:
    return " ".join(str(name).lower().replace("_", " ").split())


def _matches(name: str, fragment: str) -> bool:
    """Whole-word match for short fragments ('ph', 'do', 'od'), substring otherwise."""
    n = _norm(name)
    if len(fragment) <= 3:
        words = "".join(ch if ch.isalnum() else " " for ch in n).split()
        return fragment in words
    return fragment in n


def suggest_variable(column: str, export_names: dict[str, list[str]] | None = None) -> str | None:
    """The registered variable an export column most likely holds, or None.

    ``export_names`` (from a plant profile) are exact names checked first."""
    for var, names in (export_names or {}).items():
        if _norm(column) in {_norm(n) for n in names}:
            return var
    for var, fragments in VARIABLE_HINTS.items():
        if any(_matches(column, f) for f in fragments):
            return var
    n = _norm(column)
    return n if n in VARIABLES else None


def suggest_column(columns, field: str, exclude=()) -> str | None:
    """The column most likely to hold ``field`` (e.g. 'batch_id'), or None."""
    candidates = [c for c in columns if c not in exclude]
    for fragment in COLUMN_HINTS.get(field, (field,)):
        for c in candidates:
            if _norm(c) == fragment:
                return c
    for fragment in COLUMN_HINTS.get(field, (field,)):
        for c in candidates:
            if _matches(c, fragment):
                return c
    return None


FILE_KINDS = (
    "online signals (one column per signal)",
    "lab results (one row per result)",
    "batch records",
    "outcomes",
    "operator notes",
    "ignore",
)


def suggest_file_kind(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    if suggest_column(cols, "text") and len(cols) <= 4:
        return FILE_KINDS[4]
    if suggest_column(cols, "category") and not suggest_column(cols, "timestamp"):
        return FILE_KINDS[3]
    if suggest_column(cols, "start_time") or suggest_column(cols, "bioreactor_id"):
        return FILE_KINDS[2]
    if suggest_column(cols, "variable") and suggest_column(cols, "value"):
        return FILE_KINDS[1]
    return FILE_KINDS[0]


CATEGORY_HINTS = {  # outcome label fragments -> category
    "contamination": ("contam", "infect", "foreign"),
    "normal": ("ok", "normal", "pass", "good", "released", "approved", "in spec"),
    "non_contamination_anomaly": ("other", "fail", "deviation", "low yield", "anomal"),
}
ROLE_CANDIDATES = {  # role -> variables that can play it, in order of preference
    "biomass": ("biomass_od600", "biomass_offline"),
    "oxygen": ("dissolved_oxygen", "do2_mg_l"),
    "ph": ("ph", "ph_offline"),
    "product": ("enzyme_activity", "penicillin_offline"),
    "temperature": ("temperature",),
    "ph_control": ("base_flow",),
    "temperature_control": ("coolant_flow",),
}


def suggest_category(label) -> str | None:
    n = _norm(label)
    for category, fragments in CATEGORY_HINTS.items():
        if any(_matches(n, f) for f in fragments):
            return category
    return n if n in OUTCOME_CATEGORIES else None


def _is_datetime(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(20)
    if sample.empty or pd.to_numeric(sample, errors="coerce").notna().all():
        return False
    return pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.8


def _time_fields(df: pd.DataFrame, exclude) -> dict:
    """Timestamp column if one parses as dates, else an elapsed-hours column."""
    cols = [c for c in df.columns if c not in exclude]
    ts = suggest_column(cols, "timestamp")
    if ts and _is_datetime(df[ts]):
        return {"timestamp": ts}
    el = suggest_column(cols, "elapsed_hours")
    if el and pd.to_numeric(df[el], errors="coerce").notna().any():
        return {"elapsed_hours": el}
    return {"timestamp": ts} if ts else {}


def draft_mapping(
    frames: dict[str, pd.DataFrame],
    export_names: dict[str, list[str]] | None = None,
    process_name: str = "imported process",
    kinds: dict[str, str] | None = None,
) -> dict:
    """A suggested mapping for uploaded exports ({file name: table}), to be reviewed.

    Every field is a guess from column names and contents (``kinds`` overrides the
    guessed file kinds); the dashboard shows each one for the user to confirm or change
    before anything is converted."""
    m: dict = {"process_name": process_name, "variables": {}, "measurements": []}
    kinds = {name: (kinds or {}).get(name) or suggest_file_kind(df) for name, df in frames.items()}
    for name, df in frames.items():
        kind, cols = kinds[name], list(df.columns)
        batch = suggest_column(cols, "batch_id")
        if kind == FILE_KINDS[0]:
            spec = {"file": name, "layout": "wide", "batch_id": batch}
            spec |= _time_fields(df, {batch})
            spec["source"] = "online_sensor"
            used = {batch, *spec.values()}
            spec["columns"] = {
                c: v
                for c in cols
                if c not in used and (v := suggest_variable(c, export_names)) is not None
            }
            m["measurements"].append(spec)
        elif kind == FILE_KINDS[1]:
            spec = {"file": name, "layout": "long", "batch_id": batch}
            spec |= _time_fields(df, {batch})
            used = {batch, *spec.values()}
            for f in ("available_at", "variable", "value", "unit"):
                c = suggest_column(cols, f, exclude=used)
                if c:
                    spec[f] = c
                    used.add(c)
            spec["source"] = "offline_assay"
            names = df[spec["variable"]].dropna().astype(str).unique() if "variable" in spec else []
            spec["variable_map"] = {
                n: v for n in names if (v := suggest_variable(n, export_names)) is not None
            }
            m["measurements"].append(spec)
        elif kind == FILE_KINDS[2]:
            fields = ("start_time", *GENEALOGY_COLUMNS)
            found = {"batch_id": batch}
            for f in fields:
                c = suggest_column(cols, f, exclude=set(found.values()))
                if c:
                    found[f] = c
            m["batches"] = {"file": name, "columns": found}
        elif kind == FILE_KINDS[3]:
            found = {"batch_id": batch}
            for f in ("category", "onset_h", "reference_event_h", "reason"):
                c = suggest_column(cols, f, exclude=set(found.values()))
                if c:
                    found[f] = c
            spec = {"file": name, "columns": found}
            if "category" in found:
                labels = df[found["category"]].dropna().astype(str).unique()
                spec["category_map"] = {lab: suggest_category(lab) for lab in labels}
            m["outcomes"] = spec
        elif kind == FILE_KINDS[4]:
            spec = {"file": name, "batch_id": batch}
            spec["timestamp"] = suggest_column(cols, "timestamp", exclude={batch})
            spec["text"] = suggest_column(cols, "text", exclude={batch, spec["timestamp"]})
            m.setdefault("observations", []).append(spec)
    m["roles"] = suggest_roles(mapped_variables(m))
    return m


def mapped_variables(m: dict) -> list[str]:
    """Variables a mapping imports, in first-seen order."""
    out: list[str] = []
    for spec in m.get("measurements", []):
        for v in (*spec.get("columns", {}).values(), *spec.get("variable_map", {}).values()):
            if v and v not in out:
                out.append(v)
    return out


def suggest_roles(variables) -> dict[str, str]:
    have = set(variables)
    return {
        role: next((v for v in candidates if v in have), "not_measured")
        for role, candidates in ROLE_CANDIDATES.items()
    }


def mapping_yaml(m: dict) -> str:
    """The mapping as YAML, so it can be saved and reused from the command line."""
    return yaml.safe_dump(m, sort_keys=False, allow_unicode=True)


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
