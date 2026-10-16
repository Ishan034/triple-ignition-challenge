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
