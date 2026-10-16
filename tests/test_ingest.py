import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from contamination.ingest import validate_measurements


def _row(**overrides):
    row = {
        "batch_id": "B001",
        "timestamp": pd.Timestamp("2026-01-01 00:00"),
        "available_at": pd.Timestamp("2026-01-01 00:00"),
        "stage": "lag",
        "variable": "ph",
        "value": 7.0,
        "unit": "pH",
        "source": "online_probe",
        "quality_flag": "valid",
        "observation": None,
    }
    row.update(overrides)
    return row


def _validate(rows):
    return validate_measurements(pd.DataFrame(rows))


def test_clean_rows_pass_without_issues():
    res = _validate(
        [
            _row(),
            _row(
                timestamp=pd.Timestamp("2026-01-01 01:00"),
                available_at=pd.Timestamp("2026-01-01 01:00"),
            ),
        ]
    )
    assert res.issues.empty
    assert (res.measurements["quality_flag"] == "valid").all()
    assert res.measurements["elapsed_h"].tolist() == [0.0, 1.0]


def test_missing_value_is_flagged_not_filled():
    res = _validate([_row(value=np.nan)])
    m = res.measurements.iloc[0]
    assert m["quality_flag"] == "missing"
    assert np.isnan(m["value"])  # never filled with zero


def test_unit_mismatch_flagged_and_value_preserved():
    res = _validate([_row(variable="temperature", unit="degF", value=98.6)])
    m = res.measurements.iloc[0]
    assert m["quality_flag"] == "suspect"
    assert m["value"] == 98.6 and m["unit"] == "degF"
    assert "unit" in m["quality_note"]


def test_implausible_value_flagged_not_dropped():
    res = _validate([_row(value=15.2)])
    assert len(res.measurements) == 1
    assert res.measurements.iloc[0]["quality_flag"] == "suspect"


def test_exact_duplicate_dropped_and_reported():
    res = _validate([_row(), _row()])
    assert len(res.measurements) == 1
    assert (res.issues["issue"] == "exact duplicate row").sum() == 1


def test_conflicting_duplicates_all_flagged():
    res = _validate([_row(value=7.0), _row(value=6.5)])
    assert len(res.measurements) == 2
    assert (res.measurements["quality_flag"] == "suspect").all()


def test_available_before_measured_flagged():
    res = _validate([_row(available_at=pd.Timestamp("2025-12-31 23:00"))])
    assert res.measurements.iloc[0]["quality_flag"] == "suspect"


def test_unknown_variable_and_stage_flagged():
    res = _validate([_row(variable="glucose", unit="g/L", stage="harvest")])
    note = res.measurements.iloc[0]["quality_note"]
    assert "unknown variable" in note and "unknown stage" in note


def test_missing_required_column_raises():
    df = pd.DataFrame([_row()]).drop(columns=["timestamp"])
    with pytest.raises(pa.errors.SchemaError):
        validate_measurements(df)
