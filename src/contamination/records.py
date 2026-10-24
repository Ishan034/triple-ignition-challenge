"""What scientists record in the dashboard: check results and alert feedback.

Kept per campaign under ``data/records/<campaign>/`` (git-ignored), apart from the
campaign's own data, so recording never edits the measurements or the outcomes table.

* ``recorded_checks.csv``: check results added in the dashboard (the campaign's own
  ``investigations.csv`` is left untouched; the two are combined when loading).
* ``alert_feedback.csv``: per alert, whether it pointed at a real issue.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from .schema import INVESTIGATION_COLUMNS

RECORDS_ROOT = Path("data/records")
ENV_ROOT = "CONTAMINATION_RECORDS"  # override the location (tests, deployments)
CHECKS = "recorded_checks.csv"
FEEDBACK = "alert_feedback.csv"
FEEDBACK_COLUMNS = ["batch_id", "variable", "detector", "start_h", "verdict", "note"]
VERDICTS = ("real issue", "false alarm", "not sure")


def records_dir(data_dir: str | Path, root: Path | None = None) -> Path:
    """One folder per campaign, named after its path (``data/synthetic`` -> ``data_synthetic``)."""
    root = root or Path(os.environ.get(ENV_ROOT, RECORDS_ROOT))
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(Path(data_dir)).strip("/\\")).strip("_")
    return root / (name or "campaign")


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, dtype={"note": "object"})
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    return df[columns]


def load_checks(data_dir, root: Path | None = None) -> pd.DataFrame:
    return _read(records_dir(data_dir, root) / CHECKS, INVESTIGATION_COLUMNS)


def load_feedback(data_dir, root: Path | None = None) -> pd.DataFrame:
    return _read(records_dir(data_dir, root) / FEEDBACK, FEEDBACK_COLUMNS)


def _save(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_checks(data_dir, df: pd.DataFrame, root: Path | None = None) -> Path:
    return _save(df[INVESTIGATION_COLUMNS].drop_duplicates(), records_dir(data_dir, root) / CHECKS)


def save_feedback(data_dir, df: pd.DataFrame, root: Path | None = None) -> Path:
    bad = set(df["verdict"].dropna()) - set(VERDICTS)
    if bad:
        raise ValueError(f"unknown verdicts {sorted(bad)}; use {VERDICTS}")
    # One verdict per alert: the latest wins.
    key = ["batch_id", "variable", "detector", "start_h"]
    latest = df[FEEDBACK_COLUMNS].drop_duplicates(subset=key, keep="last")
    return _save(latest, records_dir(data_dir, root) / FEEDBACK)


def merge_upload(existing: pd.DataFrame, upload: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Restore from a downloaded CSV: rows from both, exact duplicates once."""
    missing = set(columns) - set(upload.columns)
    if missing:
        raise ValueError(f"the file is missing columns {sorted(missing)}")
    return pd.concat([existing[columns], upload[columns]], ignore_index=True).drop_duplicates()
