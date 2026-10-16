"""Diagnosis-layer tests on synthetic data (software behaviour, not real performance)."""

import pandas as pd
import pytest
import yaml

from contamination.diagnosis import CATALOGUE_PATH, assess, diagnose, load_knowledge
from contamination.evaluate import EXPECTED_HYPOTHESIS
from contamination.ingest import validate_measurements
from contamination.monitor import fit_reference
from contamination.schema import INVESTIGATION_COLUMNS
from contamination.synthetic import generate_dataset


@pytest.fixture(scope="module")
def setup():
    ds = generate_dataset(
        n_reference=20,
        n_test_normal=3,
        n_slow_growth=2,
        n_contamination=3,
        n_sensor_spike=2,
        n_do_drift=2,
        seed=5,
    )
    m = validate_measurements(ds.measurements).measurements
    ordered = ds.batches.sort_values("start_time")["batch_id"].tolist()
    reference = fit_reference(m, ordered[:20])
    test = m[m["batch_id"].isin(ordered[20:])]
    full = diagnose(test, reference)
    return ds, test, reference, full


def _scenario_batches(ds, scenario):
    return ds.outcomes.loc[ds.outcomes["scenario"] == scenario, "batch_id"].tolist()


def _inv(rows):
    return pd.DataFrame(rows, columns=INVESTIGATION_COLUMNS)


def test_knowledge_base_loads_and_is_marked_draft():
    kb = load_knowledge()
    assert "draft" in kb.review_status
    assert {"H01", "H02", "H03", "H04", "H05"} <= set(kb.hypotheses)


def test_knowledge_base_rejects_undefined_evidence(tmp_path):
    d = yaml.safe_load(CATALOGUE_PATH.read_text())
    d["hypotheses"]["H01"]["supports"]["E99"] = 1
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):
        load_knowledge(bad)


def test_end_of_batch_leading_explanations(setup):
    """Never a wrong leading explanation; "unresolved" (H05) is the allowed safe failure."""
    ds, _, _, full = setup
    table = full.table().merge(ds.outcomes[["batch_id", "scenario"]], on="batch_id")
    matched = total = 0
    for scenario, expected in EXPECTED_HYPOTHESIS.items():
        got = table.loc[table["scenario"] == scenario, "leading_id"]
        assert len(got) and got.isin([expected, "H05"]).all(), (scenario, got.tolist())
        matched += int((got == expected).sum())
        total += len(got)
    assert matched >= 0.75 * total
    contamination = table.loc[table["scenario"] == "contamination", "leading_id"]
    assert (contamination == "H01").all()


def test_never_wrong_leading_at_first_alert(setup):
    """Early on, the safe answer is 'unresolved', never a wrong explanation."""
    ds, test, reference, full = setup
    for b, r in full.reports.items():
        scenario = ds.outcomes.set_index("batch_id").loc[b, "scenario"]
        if scenario not in EXPECTED_HYPOTHESIS:
            continue
        t = min(a.first_raised_h for a in r.alerts)
        early = diagnose(test[test["batch_id"] == b], reference, as_of_h=t).reports[b]
        lead = early.leading
        assert lead is not None
        assert lead.id in (EXPECTED_HYPOTHESIS[scenario], "H05")


def test_no_contamination_claim_without_independent_test(setup):
    _, _, _, full = setup
    for r in full.reports.values():
        assert r.assessment["contamination_evidence"].startswith("unconfirmed")
        assert not r.assessment["confidence"].startswith("strong")
        text = r.to_markdown().lower()
        assert "e. coli" not in text and "coli" not in text
        assert "not probabilities" in text


def test_unknown_evidence_is_not_counted(setup):
    _, _, _, full = setup
    for r in full.reports.values():
        unknown = {e for e, item in r.evidence.items() if item.state == "unknown"}
        for h in r.hypotheses:
            assert not unknown & (set(h.supporting) | set(h.contradicting))


def test_positive_test_confirms_and_raises_confidence(setup):
    ds, test, reference, _ = setup
    b = _scenario_batches(ds, "contamination")[0]
    inv = _inv([[b, "T01", "positive", 40.0, "test"]])
    r = diagnose(test[test["batch_id"] == b], reference, investigations=inv).reports[b]
    assert r.evidence["E12"].state == "present"
    assert r.assessment["contamination_evidence"] == "confirmed by independent test"
    assert r.assessment["confidence"].startswith("strong")
    assert all(c["check_id"] != "T04" for c in r.next_checks)  # no screening once confirmed
    assert all(c["check_id"] != "T01" for c in r.next_checks)  # done already


def test_negative_test_contradicts_contamination(setup):
    ds, test, reference, full = setup
    b = _scenario_batches(ds, "contamination")[0]
    inv = _inv([[b, "T01", "negative", 40.0, None]])
    r = diagnose(test[test["batch_id"] == b], reference, investigations=inv).reports[b]
    before = next(h for h in full.reports[b].hypotheses if h.id == "H01").score
    after = next(h for h in r.hypotheses if h.id == "H01").score
    assert "E13" in next(h for h in r.hypotheses if h.id == "H01").contradicting
    assert after < before
    assert r.assessment["contamination_evidence"] == "independent test negative"


def test_investigations_after_as_of_are_ignored(setup):
    ds, test, reference, _ = setup
    b = _scenario_batches(ds, "contamination")[0]
    inv = _inv([[b, "T01", "positive", 47.0, None]])
    r = diagnose(test[test["batch_id"] == b], reference, investigations=inv, as_of_h=40.0).reports[
        b
    ]
    assert r.evidence["E12"].state == "unknown"


def test_instrument_fault_check_supports_measurement_fault(setup):
    ds, test, reference, _ = setup
    b = _scenario_batches(ds, "do_probe_drift")[0]
    inv = _inv([[b, "T02", "positive", 45.0, None]])
    r = diagnose(test[test["batch_id"] == b], reference, investigations=inv).reports[b]
    assert r.leading.id == "H04"
    assert "E15" in r.leading.supporting


def test_safety_rule_keeps_contamination_check(setup):
    """While contamination is not contradicted, a check informative about it is recommended."""
    _, _, _, full = setup
    kb = load_knowledge()
    for r in full.reports.values():
        h01 = next(h for h in r.hypotheses if h.id == "H01")
        if h01.score >= 0:
            assert any("H01" in kb.checks[c["check_id"]]["discriminates"] for c in r.next_checks)


def test_single_weak_item_cannot_lead():
    kb = load_knowledge()
    from contamination.diagnosis import EvidenceItem

    evidence = {e: EvidenceItem(e, kb.evidence[e]["name"], "absent", "") for e in kb.evidence}
    evidence["E02"].state = "present"  # weight 1 for H04 only
    hyps = assess(evidence, kb)
    assert hyps[0].id == "H04" and hyps[0].tier == "possible"
    assert next(h for h in hyps if h.id == "H05").tier == "leading"


def test_report_markdown_sections(setup):
    _, _, _, full = setup
    text = next(iter(full.reports.values())).to_markdown()
    for heading in (
        "What changed?",
        "Plausible explanations",
        "missing",
        "Assessment",
        "next checks",
        "Scientist decision",
    ):
        assert heading.lower() in text.lower()


def test_investigations_for_other_batches_are_ignored(setup):
    ds, test, reference, _ = setup
    b, other = _scenario_batches(ds, "contamination")[:2]
    inv = _inv([[other, "T01", "positive", 30.0, None]])
    reports = diagnose(test[test["batch_id"] == b], reference, investigations=inv).reports
    assert set(reports) == {b}
