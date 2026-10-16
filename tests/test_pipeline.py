"""End-to-end tests on a small synthetic dataset (software behaviour, not real performance)."""

import numpy as np
import pandas as pd
import pytest

from contamination.evaluate import evaluate
from contamination.ingest import validate_measurements
from contamination.monitor import MonitorConfig, fit_reference, monitor
from contamination.reference import ReferenceProfile, build_reference
from contamination.synthetic import generate_dataset


@pytest.fixture(scope="module")
def setup():
    ds = generate_dataset(
        n_reference=15,
        n_test_normal=4,
        n_slow_growth=1,
        n_contamination=3,
        n_sensor_spike=1,
        n_do_drift=1,
        seed=11,
    )
    m = validate_measurements(ds.measurements).measurements
    ordered = ds.batches.sort_values("start_time")["batch_id"].tolist()
    ref_ids, test_ids = ordered[:15], ordered[15:]
    reference = fit_reference(m, ref_ids, MonitorConfig())
    result = monitor(m[m["batch_id"].isin(test_ids)], reference)
    return ds, m, ref_ids, test_ids, reference, result


def test_synthetic_reference_batches_are_all_normal(setup):
    ds, _, ref_ids, *_ = setup
    assert (ds.outcomes.set_index("batch_id").loc[ref_ids, "category"] == "normal").all()


def test_reference_built_only_from_approved_batches(setup):
    _, _, ref_ids, test_ids, reference, _ = setup
    assert set(reference.batch_ids) == set(ref_ids)
    assert not set(reference.batch_ids) & set(test_ids)


def test_reference_rejects_unknown_batches(setup):
    _, m, *_ = setup
    with pytest.raises(ValueError):
        build_reference(m, ["NOPE"])


def test_mad_zero_falls_back_to_min_scale():
    rows = []
    for b in range(5):
        for h in range(3):
            t = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=h)
            rows.append(
                {
                    "batch_id": f"B{b}",
                    "timestamp": t,
                    "available_at": t,
                    "stage": "lag",
                    "variable": "temperature",
                    "value": 37.0,
                    "unit": "degC",
                    "source": "probe",
                    "quality_flag": "valid",
                    "observation": None,
                }
            )
    m = validate_measurements(pd.DataFrame(rows)).measurements
    ref = build_reference(m, [f"B{b}" for b in range(5)])
    assert (ref.stats["mad"] == 0).all()
    assert (ref.stats["scale"] == 0.05).all()


def test_reference_roundtrip_and_tamper_check(setup, tmp_path):
    *_, reference, _ = setup
    path = tmp_path / "ref.json"
    reference.save(path)
    loaded = ReferenceProfile.load(path)
    assert loaded.version == reference.version
    assert loaded.thresholds == reference.thresholds
    tampered = path.read_text().replace('"window_h": 1.0', '"window_h": 2.0')
    path.write_text(tampered)
    with pytest.raises(ValueError):
        ReferenceProfile.load(path)


def test_alerts_are_structured_and_never_claim_contamination(setup):
    *_, reference, result = setup
    assert result.alerts
    for a in result.alerts:
        assert a.contamination_confirmed is False
        assert a.reference_version == reference.version
        assert a.first_raised_h >= a.start_h  # cannot be raised before the result existed
        assert a.persistence in {"isolated", "persistent"}
        assert "not a contamination probability" in a.interpretation.lower()
    assert (result.summary["contamination_evidence"].str.startswith("none recorded")).all()


def test_contaminated_batches_flagged_after_onset(setup):
    ds, _, _, test_ids, _, result = setup
    outcomes = ds.outcomes[ds.outcomes["batch_id"].isin(test_ids)]
    metrics = evaluate(result.alerts, outcomes)
    assert metrics["sensitivity"] == 1.0
    assert metrics["median_onset_to_alert_h"] > 0


def test_monitoring_is_deterministic(setup):
    _, m, _, test_ids, reference, result = setup
    again = monitor(m[m["batch_id"].isin(test_ids)], reference)
    assert [a.to_dict() for a in again.alerts] == [a.to_dict() for a in result.alerts]


def test_as_of_uses_only_available_results(setup):
    _, m, _, test_ids, reference, _result = setup
    as_of = 20.0
    early = monitor(m[m["batch_id"].isin(test_ids)], reference, as_of_h=as_of)
    assert (early.scores["available_h"] <= as_of).all()
    assert all(a.first_raised_h <= as_of for a in early.alerts)


def test_missing_readings_excluded_from_scores(setup):
    _, m, _, test_ids, _, result = setup
    missing = m[(m["quality_flag"] == "missing") & m["batch_id"].isin(test_ids)]
    assert not result.scores["value"].isna().any()
    if len(missing):
        keys = set(zip(missing["batch_id"], missing["variable"], missing["elapsed_h"]))
        scored = set(
            zip(result.scores["batch_id"], result.scores["variable"], result.scores["elapsed_h"])
        )
        assert not keys & scored


def test_normal_test_batches_mostly_quiet(setup):
    ds, _, _, test_ids, _, result = setup
    normal = ds.outcomes[
        (ds.outcomes["scenario"] == "normal") & ds.outcomes["batch_id"].isin(test_ids)
    ]
    flagged = {a.batch_id for a in result.alerts}
    assert np.mean([b in flagged for b in normal["batch_id"]]) <= 0.5
