"""Layer 3a, reference memory: what a normal batch looks like.

Statistical process monitoring has two phases (NIST):

* Phase I builds the reference from batches the scientists approve as normal.
  :func:`build_reference` takes an explicit list of approved batch IDs and never
  sees the outcomes table.
* Phase II monitors new batches against that reference, which stays *frozen*. A
  :class:`ReferenceProfile` carries a version hash and the list of batches it was
  built from, and is only replaced through a new, documented build. It is never
  updated automatically.

For each variable and each elapsed hour, the reference stores the median and a
robust scale (1.4826 x MAD, which matches the standard deviation for normal data).
Values within +/- ``window_h`` hours are pooled to steady the estimates when there
are few batches. When MAD is 0, the scale falls back to the variable's ``min_scale``
instead of dividing by zero.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import valid_values
from .schema import VARIABLES

MAD_TO_SIGMA = 1.4826  # 1 / 0.6745

STATS_COLUMNS = ["variable", "hour", "median", "scale", "mad", "n_values", "n_batches"]


@dataclass(frozen=True)
class ReferenceProfile:
    stats: pd.DataFrame
    batch_ids: tuple[str, ...]
    params: dict
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    created_at: str = ""
    version: str = ""

    def lookup(self, variable: pd.Series, elapsed_h: pd.Series) -> pd.DataFrame:
        """Reference median/scale/support for each (variable, elapsed hour) pair."""
        keys = pd.DataFrame(
            {"variable": variable.to_numpy(), "hour": np.rint(elapsed_h.to_numpy())}
        )
        merged = keys.merge(self.stats, on=["variable", "hour"], how="left")
        return merged[["median", "scale", "n_batches"]].set_axis(variable.index)

    def with_thresholds(self, thresholds: dict, calibration: dict) -> ReferenceProfile:
        frozen = replace(
            self,
            thresholds=thresholds,
            calibration=calibration,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        return replace(frozen, version=_version(frozen))

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "batch_ids": list(self.batch_ids),
            "params": self.params,
            "thresholds": self.thresholds,
            "calibration": self.calibration,
            "stats": self.stats.to_dict(orient="records"),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ReferenceProfile:
        d = json.loads(Path(path).read_text())
        profile = cls(
            stats=pd.DataFrame(d["stats"], columns=STATS_COLUMNS),
            batch_ids=tuple(d["batch_ids"]),
            params=d["params"],
            thresholds=d["thresholds"],
            calibration=d["calibration"],
            created_at=d["created_at"],
            version=d["version"],
        )
        if _version(profile) != profile.version:
            raise ValueError("Reference profile contents do not match its version hash")
        return profile


def _version(profile: ReferenceProfile) -> str:
    payload = json.dumps(
        {
            "batch_ids": sorted(profile.batch_ids),
            "params": profile.params,
            "thresholds": profile.thresholds,
            "stats": profile.stats.round(6).to_dict(orient="records"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def build_reference(
    measurements: pd.DataFrame,
    batch_ids,
    window_h: float = 1.0,
    control_loops: dict | None = None,
    linked: dict | None = None,
) -> ReferenceProfile:
    """Phase I: estimate robust per-hour reference statistics from approved batches."""
    batch_ids = tuple(sorted(set(batch_ids)))
    unknown = set(batch_ids) - set(measurements["batch_id"])
    if unknown:
        raise ValueError(f"Approved batches not found in measurements: {sorted(unknown)}")
    data = valid_values(measurements[measurements["batch_id"].isin(batch_ids)])
    data = data[data["variable"].isin(VARIABLES.keys())]

    rows = []
    for variable, group in data.groupby("variable"):
        elapsed = group["elapsed_h"].to_numpy()
        values = group["value"].to_numpy()
        batches = group["batch_id"].to_numpy()
        for hour in np.unique(np.rint(elapsed)):
            sel = np.abs(elapsed - hour) <= window_h
            v = values[sel]
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)))
            scale = max(MAD_TO_SIGMA * mad, VARIABLES[variable].min_scale)
            rows.append(
                [variable, float(hour), med, scale, mad, int(sel.sum()), len(set(batches[sel]))]
            )

    stats = pd.DataFrame(rows, columns=STATS_COLUMNS)
    params: dict = {"window_h": window_h}
    if control_loops:
        # {controlled variable: [controller output variable, output raises the variable]}
        params["control_loops"] = {c: [e, bool(r)] for c, (e, r) in sorted(control_loops.items())}
    if linked:
        # {"signals": [...], "scales": {signal: normal spread of its smoothed change}}
        params["linked_shift"] = linked
    return ReferenceProfile(stats=stats, batch_ids=batch_ids, params=params)
