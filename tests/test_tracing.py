import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

from contamination.diagnosis import load_knowledge
from contamination.evaluate import evaluate_tracing
from contamination.tracing import _relative_risk, genealogy_graph, trace_sources


def _batches(n=24):
    rows = []
    for i in range(n):
        rows.append(
            {
                "batch_id": f"B{i:02d}",
                "start_time": pd.Timestamp("2026-01-01") + pd.Timedelta(days=2 * i),
                "bioreactor_id": f"BR-{i % 2 + 1}",
                "medium_lot": f"M-{i // 3:02d}",
                "inoculum_lot": f"I-{i // 6:02d}",
            }
        )
    return pd.DataFrame(rows)


def _statuses(batches, cases, uncertain=()):
    rows = []
    for b in batches["batch_id"]:
        if b in cases:
            rows.append((b, "contamination suspected", True))
        elif b in uncertain:
            rows.append((b, "unresolved", None))
        else:
            rows.append((b, "approved normal (reference batch)", False))
    return pd.DataFrame(rows, columns=["batch_id", "status", "is_case"])


@pytest.fixture
def planted():
    """Batches B03-B05 share medium lot M-01; a fourth case B20 is independent."""
    b = _batches()
    b.loc[b["batch_id"].isin(["B03", "B04", "B05"]), "medium_lot"] = "M-XX"
    return b, ["B03", "B04", "B05", "B20"]


def test_genealogy_graph_links_batches_to_resources(planted):
    batches, _ = planted
    g = genealogy_graph(batches, load_knowledge().source_tracing["resources"])
    assert set(g.neighbors("B03")) == {"medium lot M-XX", "inoculum lot I-00", "bioreactor BR-2"}
    assert set(g.neighbors("medium lot M-XX")) == {"B03", "B04", "B05"}


def test_planted_lot_is_the_only_strong_lead(planted):
    batches, cases = planted
    trace = trace_sources(batches, _statuses(batches, cases))
    strong = [c.resource for c in trace.candidates if c.tier == "strong lead"]
    assert strong == ["medium lot M-XX"]
    assert trace.candidates[0].resource == "medium lot M-XX"


def test_fisher_p_value_and_bonferroni(planted):
    batches, cases = planted
    trace = trace_sources(batches, _statuses(batches, cases))
    lot = next(c for c in trace.candidates if c.resource == "medium lot M-XX")
    # 3 of 3 users are cases; 1 of 21 non-users is a case.
    expected = fisher_exact([[3, 0], [1, 20]], alternative="greater")[1]
    assert lot.p_value == pytest.approx(expected)
    n_tests = sum(1 for c in trace.candidates if len(c.cases) >= 2)
    assert lot.p_adjusted == pytest.approx(min(1.0, expected * n_tests))


def test_bioreactor_shared_by_cases_is_not_a_lead(planted):
    batches, cases = planted
    trace = trace_sources(batches, _statuses(batches, cases))
    br = [c for c in trace.candidates if c.resource_type == "bioreactor"]
    assert all(c.tier in ("not over-represented", "single case", "possible lead") for c in br)
    assert all(c.tier != "strong lead" for c in br)


def test_independent_case_has_no_lead(planted):
    batches, cases = planted
    trace = trace_sources(batches, _statuses(batches, cases))
    assert trace.per_case["B20"] == []
    assert any(set(c["cases"]) == {"B03", "B04", "B05"} for c in trace.clusters)


def test_uncertain_batches_are_excluded(planted):
    batches, cases = planted
    trace = trace_sources(batches, _statuses(batches, cases, uncertain=["B10"]))
    m = next(c for c in trace.candidates if c.resource == "medium lot M-XX")
    assert m.cases_elsewhere + m.non_cases_elsewhere == 20  # 24 - 3 users - 1 uncertain


def test_numpy_bool_statuses_are_handled(planted):
    batches, cases = planted
    st = _statuses(batches, cases)
    st["is_case"] = st["is_case"].map(lambda v: np.bool_(v) if v is not None else None)
    trace = trace_sources(batches, st)
    assert trace.candidates[0].resource == "medium lot M-XX"


def test_relative_risk_with_zero_cell_uses_correction():
    assert np.isfinite(_relative_risk(3, 0, 1, 20))
    assert _relative_risk(2, 2, 2, 2) == pytest.approx(1.0)


def test_evaluate_tracing(planted):
    batches, cases = planted
    trace = trace_sources(batches, _statuses(batches, cases))
    outcomes = pd.DataFrame(
        {
            "batch_id": cases,
            "simulated_source": ["medium lot M-XX"] * 3 + ["independent (not lot-related)"],
        }
    )
    ev = evaluate_tracing(trace, outcomes)
    assert ev["planted_tier"] == "strong lead" and ev["planted_is_top_lead"]
    assert ev["planted_cases_linked"] == "3/3"
    assert ev["independent_cases_without_strong_lead"] == "1/1"
    assert ev["false_strong_leads"] == ""


def test_end_to_end_affected_scope_on_synthetic_data():
    from contamination.diagnosis import diagnose
    from contamination.ingest import validate_measurements
    from contamination.monitor import fit_reference
    from contamination.synthetic import generate_dataset
    from contamination.tracing import apply_to_reports, batch_statuses

    ds = generate_dataset(seed=7)
    m = validate_measurements(ds.measurements).measurements
    ordered = ds.batches.sort_values("start_time")["batch_id"].tolist()
    ref = fit_reference(m, ordered[:30])
    diag = diagnose(m[m["batch_id"].isin(ordered[30:])], ref)
    statuses = batch_statuses(diag, ordered[30:], ordered[:30])
    trace = trace_sources(ds.batches, statuses)
    apply_to_reports(diag, trace)
    planted = ds.outcomes.loc[ds.outcomes["simulated_source"] == "medium lot M-12", "batch_id"]
    for b in planted:
        assert "medium lot M-12" in diag.reports[b].assessment["affected_scope"]
        assert any(s["resource"] == "medium lot M-12" for s in diag.reports[b].sources)
        assert "Possible sources" in diag.reports[b].to_markdown()
    assert evaluate_tracing(trace, ds.outcomes)["false_strong_leads"] == ""
