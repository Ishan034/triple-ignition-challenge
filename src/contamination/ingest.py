"""Layer 1, evidence acquisition: load and validate batch measurements.

Rules (docs/project-context.md, "Ingestion rules"):

* Structural problems (missing columns, unparseable timestamps) raise an error.
* Value-level problems are *flagged*, not deleted: the row's ``quality_flag`` becomes
  ``suspect`` or ``missing`` and the reason is written to ``quality_note`` and to the
  issues table. The original value and unit are kept unchanged.
* Missing values are never filled with zero.
* Exact duplicate rows are dropped (and reported); conflicting duplicates are all
  kept and marked suspect for review.
* ``available_h`` (when a result became available) is kept beside ``elapsed_h`` so
  later analysis cannot use results before they existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    BATCH_SCHEMA,
    INVESTIGATION_SCHEMA,
    MEASUREMENT_SCHEMA,
    OUTCOME_SCHEMA,
    QUALITY_FLAGS,
    STAGES,
    VARIABLES,
)

ISSUE_COLUMNS = ["batch_id", "variable", "timestamp", "issue", "action"]
KEY = ["batch_id", "timestamp", "variable"]


@dataclass
class IngestResult:
    measurements: pd.DataFrame
    issues: pd.DataFrame

    def summary(self) -> pd.Series:
        return self.issues.groupby("issue").size().sort_values(ascending=False)


def _flag(df: pd.DataFrame, mask: pd.Series, flag: str, note: str, issues: list) -> None:
    if not mask.any():
        return
    df.loc[mask, "quality_flag"] = flag
    existing = df.loc[mask, "quality_note"]
    df.loc[mask, "quality_note"] = np.where(existing == "", note, existing + "; " + note)
    for row in df.loc[mask, ["batch_id", "variable", "timestamp"]].itertuples(index=False):
        issues.append([row.batch_id, row.variable, row.timestamp, note, f"marked {flag}"])


def validate_measurements(raw: pd.DataFrame, batch_start: dict | None = None) -> IngestResult:
    """Validate a measurements table and return the flagged table plus an issue report."""
    df = MEASUREMENT_SCHEMA.validate(raw.copy())
    df["quality_flag"] = df["quality_flag"].fillna("valid").astype(str).str.strip().str.lower()
    df["observation"] = df["observation"].where(df["observation"].notna(), None)
    df["quality_note"] = ""
    issues: list[list] = []

    # Exact duplicates: drop the copies, report them.
    dup_exact = df.duplicated(keep="first")
    for row in df.loc[dup_exact, ["batch_id", "variable", "timestamp"]].itertuples(index=False):
        issues.append(
            [row.batch_id, row.variable, row.timestamp, "exact duplicate row", "dropped copy"]
        )
    df = df.loc[~dup_exact].reset_index(drop=True)

    _flag(
        df, ~df["quality_flag"].isin(QUALITY_FLAGS), "suspect", "unrecognised quality flag", issues
    )
    _flag(
        df,
        df["value"].isna() & (df["quality_flag"] != "missing"),
        "missing",
        "value is empty",
        issues,
    )
    _flag(df, ~df["variable"].isin(VARIABLES.keys()), "suspect", "unknown variable", issues)
    _flag(df, df["stage"].notna() & ~df["stage"].isin(STAGES), "suspect", "unknown stage", issues)

    expected_unit = df["variable"].map({k: v.unit for k, v in VARIABLES.items()})
    _flag(
        df,
        expected_unit.notna() & (df["unit"] != expected_unit),
        "suspect",
        "unit does not match expected unit (value not converted)",
        issues,
    )

    lo = df["variable"].map({k: v.plausible_range[0] for k, v in VARIABLES.items()})
    hi = df["variable"].map({k: v.plausible_range[1] for k, v in VARIABLES.items()})
    implausible = df["value"].notna() & lo.notna() & ((df["value"] < lo) | (df["value"] > hi))
    _flag(df, implausible, "suspect", "value outside plausible range", issues)

    _flag(
        df,
        df["available_at"] < df["timestamp"],
        "suspect",
        "available before it was measured",
        issues,
    )

    conflicting = df.duplicated(subset=KEY, keep=False)
    _flag(df, conflicting, "suspect", "conflicting duplicate measurement", issues)

    # Elapsed time counts from the batch's recorded start when given (plant logs often
    # begin hours after inoculation), otherwise from its first measurement.
    first = df.groupby("batch_id")["timestamp"].transform("min")
    start = df["batch_id"].map(batch_start).fillna(first) if batch_start else first
    df["elapsed_h"] = (df["timestamp"] - start) / pd.Timedelta(hours=1)
    df["available_h"] = (df["available_at"] - start) / pd.Timedelta(hours=1)
    df = df.sort_values(["batch_id", "variable", "timestamp"]).reset_index(drop=True)

    issue_df = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    return IngestResult(df, issue_df)


def load_measurements(path: str | Path, batch_start: dict | None = None) -> IngestResult:
    raw = pd.read_csv(
        path, parse_dates=["timestamp", "available_at"], dtype={"observation": "object"}
    )
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    return validate_measurements(raw, batch_start)


def load_outcomes(path: str | Path) -> pd.DataFrame:
    """Batch outcomes: used only for evaluation, never passed to the detectors."""
    df = pd.read_csv(path)
    df["onset_h"] = pd.to_numeric(df["onset_h"], errors="coerce").astype(float)
    df["reference_event_h"] = pd.to_numeric(df["reference_event_h"], errors="coerce").astype(float)
    return OUTCOME_SCHEMA.validate(df)


def load_batches(path: str | Path) -> pd.DataFrame:
    return BATCH_SCHEMA.validate(pd.read_csv(path, parse_dates=["start_time"]))


def load_investigations(path: str | Path) -> pd.DataFrame:
    """Scientist-recorded check results (independent evidence for diagnosis)."""
    return INVESTIGATION_SCHEMA.validate(pd.read_csv(path, dtype={"note": "object"}))


def valid_values(measurements: pd.DataFrame) -> pd.DataFrame:
    """Rows usable as evidence: valid flag and a numeric value."""
    m = measurements
    return m[(m["quality_flag"] == "valid") & np.isfinite(m["value"])]
