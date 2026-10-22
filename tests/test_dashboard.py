"""Smoke tests: every dashboard page renders without errors on the synthetic campaign."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "dashboard.py")
DATA = str(ROOT / "data" / "synthetic")

pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "synthetic" / "measurements.csv").exists(),
    reason="synthetic data not generated",
)


def _app(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.text_input[0].set_value(DATA)
    at.run()
    at.sidebar.radio[0].set_value(page)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_overview_page():
    at = _app("Campaign overview")
    labels = [m.label for m in at.metric]
    assert "Flagged for review" in labels
    assert at.dataframe


def test_batch_page_and_replay():
    at = _app("Batch investigation")
    assert any(c.value == "Leading explanation" for c in at.caption)
    at.toggle[0].set_value(True)  # replay
    at.run()
    assert not at.exception
    at.slider[0].set_value(20.0)
    at.run()
    assert not at.exception


def test_tracing_page():
    at = _app("Source tracing")
    assert any("Cluster" in md.value for md in at.markdown)


def test_about_page():
    at = _app("How it works")
    assert any("Limits" in h.value for h in at.subheader)


def test_batch_page_development_and_feedback():
    at = _app("Batch investigation")
    assert any("first leading at (h)" in str(df.value.columns.tolist()) for df in at.dataframe)
    verdict = next(r for r in at.radio if r.label == "Verdict")
    verdict.set_value("false alarm")
    next(t for t in at.text_input if t.label.startswith("Note (")).set_value("probe recalibrated")
    next(b for b in at.button if b.label == "Save feedback").click()
    at.run()
    assert not at.exception
    assert any("false alarm" in str(df.value.to_dict()) for df in at.dataframe)


def test_timing_page():
    at = _app("Detection timing")
    labels = [m.label for m in at.metric]
    assert "Median lead over today's method" in labels


def test_value_page():
    at = _app("Value estimate")
    assert any(m.label.startswith("Net value per year") for m in at.metric)


def test_load_page():
    at = _app("Load your data")
    assert any(h.value == "Load your data" for h in at.header)
