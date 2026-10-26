"""Review priorities, at-risk batches and the escalation summary (synthetic campaign)."""

from pathlib import Path

import pandas as pd
import pytest

from contamination.monitor import DEFAULT_CONFIG
from contamination.pipeline import fit_cached, load_campaign, run_campaign
from contamination.triage import PRIORITIES, at_risk, escalation_text, headline, triage

pytestmark = pytest.mark.skipif(
    not Path("data/synthetic/measurements.csv").exists(), reason="synthetic data not generated"
)


@pytest.fixture(scope="module")
def campaign_result():
    camp = load_campaign("data/synthetic", 30)
    return camp, run_campaign(camp, fit_cached(camp, DEFAULT_CONFIG))


def test_priorities_follow_contamination_evidence_first(campaign_result):
    camp, res = campaign_result
    t = triage(res.overview(camp), res.trace)
    assert list(t["priority"].unique()) == [p for p in PRIORITIES if p in set(t["priority"])]
    act = t[t["priority"] == "act now"]
    assert len(act) and act["status"].str.startswith("contamination").all()
    assert not t.loc[t["priority"] == "review", "status"].str.contains("suspected").any()
    n, text = headline(t)
    assert n == int((t["priority"] != "no action").sum()) and "act now" in text


def test_at_risk_batches_used_a_lead_and_are_not_cases(campaign_result):
    _, res = campaign_result
    risk = at_risk(res.trace)
    leads = [c for c in res.trace.candidates if c.tier == "strong lead"]
    assert leads and risk
    cases = {b for c in leads for b in c.cases}
    assert not (set(risk) & cases)


def test_open_actions_are_counted(campaign_result):
    camp, res = campaign_result
    ov = res.overview(camp)
    b = ov["batch_id"].iloc[0]
    actions = pd.DataFrame(
        [[1, b, "escalate", "", "now", "open"], [2, b, "request check", "", "now", "done"]],
        columns=["action_id", "batch_id", "action", "detail", "created_at", "status"],
    )
    t = triage(ov, res.trace, actions).set_index("batch_id")
    assert t.loc[b, "open_actions"] == 1


def test_escalation_text_keeps_evidence_and_deviation_apart(campaign_result):
    _, res = campaign_result
    b, report = next(iter(res.diagnosis.reports.items()))
    text = escalation_text(b, report, "Demo plant")
    assert "Contamination evidence:" in text and "Process deviation:" in text
    assert "not probabilities" in text
