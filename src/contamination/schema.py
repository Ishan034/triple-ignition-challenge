"""Batch data model: variables, allowed codes and table schemas.

Three tables are kept separate (see docs/project-context.md):

* measurements: one row per measurement (what the detector sees)
* outcomes: one row per batch with the reviewed conclusion (never given to the detector)
* batches: one row per batch with genealogy (inputs, equipment), for later source tracing
"""

from __future__ import annotations

from dataclasses import dataclass

import pandera.pandas as pa


@dataclass(frozen=True)
class VariableSpec:
    unit: str
    # Smallest robust scale used when the reference spread is ~0 (MAD = 0 fallback).
    # Roughly the measurement resolution / noise floor of the instrument.
    min_scale: float
    # Physically plausible range; values outside are flagged "suspect", never deleted.
    plausible_range: tuple[float, float]
    description: str


# Example variables for the alpha-amylase proxy process. These are prototype
# choices, not a description of MaverickX's actual instruments.
VARIABLES: dict[str, VariableSpec] = {
    "ph": VariableSpec("pH", 0.02, (0.0, 14.0), "Culture pH"),
    "temperature": VariableSpec("degC", 0.05, (0.0, 100.0), "Culture temperature"),
    "dissolved_oxygen": VariableSpec("%sat", 1.0, (0.0, 150.0), "Dissolved oxygen"),
    "biomass_od600": VariableSpec("OD600", 0.05, (0.0, 200.0), "Optical density (biomass)"),
    "enzyme_activity": VariableSpec("U/mL", 2.0, (0.0, 1e6), "Alpha-amylase activity"),
}

STAGES = ("lag", "exponential", "stationary")
QUALITY_FLAGS = ("valid", "missing", "suspect")
OUTCOME_STATUSES = ("unknown", "reviewed", "confirmed")
OUTCOME_CATEGORIES = ("normal", "contamination", "non_contamination_anomaly")

MEASUREMENT_COLUMNS = [
    "batch_id",
    "timestamp",  # when the sample/reading was taken
    "available_at",  # when the result became available (assays can lag)
    "stage",
    "variable",
    "value",
    "unit",
    "source",
    "quality_flag",
    "observation",
]

# Structural schema only: columns must exist and be parseable. Value-level problems
# (unknown variable, wrong unit, implausible value, duplicates) are *flagged* by
# contamination.ingest rather than rejected here, so no evidence is silently lost.
MEASUREMENT_SCHEMA = pa.DataFrameSchema(
    {
        "batch_id": pa.Column(str, nullable=False),
        "timestamp": pa.Column("datetime64[ns]", nullable=False),
        "available_at": pa.Column("datetime64[ns]", nullable=False),
        "stage": pa.Column(str, nullable=True),
        "variable": pa.Column(str, nullable=False),
        "value": pa.Column(float, nullable=True),
        "unit": pa.Column(str, nullable=True),
        "source": pa.Column(str, nullable=True),
        "quality_flag": pa.Column(str, nullable=True),
        "observation": pa.Column(str, nullable=True),
    },
    coerce=True,
    strict=False,
)

OUTCOME_SCHEMA = pa.DataFrameSchema(
    {
        "batch_id": pa.Column(str, unique=True),
        "category": pa.Column(str, pa.Check.isin(OUTCOME_CATEGORIES)),
        "scenario": pa.Column(str),
        "outcome_status": pa.Column(str, pa.Check.isin(OUTCOME_STATUSES)),
        "onset_h": pa.Column(float, nullable=True),
        "reference_event_h": pa.Column(float, nullable=True),
        "reason": pa.Column(str),
    },
    coerce=True,
    strict=False,
)

BATCH_SCHEMA = pa.DataFrameSchema(
    {
        "batch_id": pa.Column(str, unique=True),
        "start_time": pa.Column("datetime64[ns]"),
        "bioreactor_id": pa.Column(str),
        "medium_lot": pa.Column(str),
        "inoculum_lot": pa.Column(str),
    },
    coerce=True,
    strict=False,
)
