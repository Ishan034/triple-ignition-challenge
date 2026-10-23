"""Limit calibration: leave-one-out quantiles and the batch-level false-alarm budget."""

from dataclasses import replace

import pytest

from contamination.ingest import validate_measurements
from contamination.monitor import DEFAULT_CONFIG, fit_reference
from contamination.synthetic import generate_dataset
from contamination.value import ValueInputs, estimate


@pytest.fixture(scope="module")
def data():
    ds = generate_dataset(
        n_reference=20, n_test_normal=0, n_slow_growth=0, n_contamination=0,
        n_sensor_spike=0, n_do_drift=0, seed=3,
    )  # fmt: skip
    m = validate_measurements(ds.measurements).measurements
    return m, sorted(m["batch_id"].unique())


def test_max_calibration_lets_no_approved_batch_alert(data):
    m, ids = data
    ref = fit_reference(m, ids, replace(DEFAULT_CONFIG, false_alarm_budget=None))
    assert ref.calibration["approved_batches_alerting"] == []


def test_budget_tightens_limits_within_the_budget(data):
    m, ids = data
    loose = fit_reference(m, ids, replace(DEFAULT_CONFIG, false_alarm_budget=None))
    tight = fit_reference(m, ids, replace(DEFAULT_CONFIG, false_alarm_budget=0.1))
    assert len(tight.calibration["approved_batches_alerting"]) <= 0.1 * len(ids)
    assert tight.calibration["quantiles"]["robust_threshold"] <= 1.0
    for v, limit in tight.thresholds["robust_threshold"].items():
        assert limit <= loose.thresholds["robust_threshold"][v]
    assert tight.thresholds["ewma"] == loose.thresholds["ewma"]  # trend detectors unchanged


def test_value_estimate_is_plain_arithmetic():
    v = ValueInputs(
        batches_per_year=100, contamination_rate=0.05, lost_batch_cost=1000,
        running_cost_per_hour=10, hours_earlier=5, share_from_shared_source=0.5,
        later_batches_saved_per_source=2, tests_avoided_per_case=1, test_cost=100,
        good_batches_saved_per_year=1, false_alarm_share=0.1, cost_per_false_alarm=20,
    )  # fmt: skip
    t = estimate(v).set_index("mechanism")["value_per_year"]
    assert t["Later batches spared by finding a shared source"] == pytest.approx(5 * 0.5 * 2 * 1000)
    assert t["Doomed batches stopped earlier"] == pytest.approx(5 * 5 * 10)
    assert t["Cost of checking false alarms"] == pytest.approx(-95 * 0.1 * 20)
    assert t["Net value per year"] == pytest.approx(t.drop("Net value per year").sum())


def test_uncertainty_ranges():
    from contamination.uncertainty import median_interval, rate, wilson

    lo, hi = wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.93 < hi < 0.95
    assert wilson(0, 40)[0] == 0.0 and wilson(10, 10)[1] == pytest.approx(1.0)
    assert rate(8, 10) == "8/10 (49–94%)"
    m, lo, hi = median_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m == 3.0 and lo <= m <= hi
