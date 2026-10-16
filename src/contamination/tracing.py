"""Source tracing: which shared inputs or equipment deserve investigation first.

The batch genealogy is a graph linking each batch to the resources it used (medium
lot, inoculum lot, bioreactor). Given each batch's contamination status (from its
diagnosis), every shared resource is compared between batches that used it and
batches that did not:

* ``attack rate``: the share of the resource's classified users that are cases.
* Fisher's exact test (one-sided: "cases are over-represented among users"), with a
  Bonferroni correction over all resources that have at least ``min_cases`` cases.
* ``tier``: "strong lead" (enough cases, significant after correction, high attack
  rate), "possible lead" (shared by >= ``min_cases`` cases, with a relative risk of
  at least ``possible_min_relative_risk``), "not over-represented" (shared, but cases are no more common among
  its users than elsewhere) or "single case".

A shared resource is a lead for investigation, never proof of cause. Cases that share
no resource with other cases are reported as possibly independent. Batches whose
status is uncertain are left out of the statistics, so they count neither for nor
against a resource.

Statuses come from diagnosis reports, so tracing uses whatever those reports saw.
For a real-time view, build the reports with ``as_of_h``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .diagnosis import DiagnosisResult, Knowledge, load_knowledge

TIER_ORDER = {"strong lead": 0, "possible lead": 1, "not over-represented": 2, "single case": 3}


def _case(value) -> bool | None:
    """Normalise a status flag (pandas may hand back numpy bools or NaN)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return bool(value)


# --------------------------------------------------------------------------- genealogy


def genealogy_graph(batches: pd.DataFrame, resources: dict[str, str]) -> nx.Graph:
    """Bipartite graph: batch nodes connected to the resource nodes they used."""
    g = nx.Graph()
    for row in batches.itertuples(index=False):
        g.add_node(row.batch_id, kind="batch", start_time=row.start_time)
        for column, label in resources.items():
            value = getattr(row, column)
            if pd.isna(value) or value == "":
                continue
            node = f"{label} {value}"
            g.add_node(node, kind="resource", resource_type=label, value=str(value))
            g.add_edge(row.batch_id, node)
    return g


# --------------------------------------------------------------------------- statuses


def batch_statuses(
    diagnosis: DiagnosisResult,
    monitored_ids,
    reference_ids=(),
) -> pd.DataFrame:
    """Contamination status per batch: case (True), non-case (False) or uncertain (None)."""
    rows = []
    for b in reference_ids:
        rows.append((b, "approved normal (reference batch)", False))
    for b in monitored_ids:
        r = diagnosis.reports.get(b)
        if r is None:
            rows.append((b, "within reference (no alerts)", False))
            continue
        lead = r.leading
        h01 = next(h for h in r.hypotheses if h.id == "H01")
        if r.evidence.get("E12") and r.evidence["E12"].state == "present":
            rows.append((b, "contamination confirmed by test", True))
        elif r.evidence.get("E13") and r.evidence["E13"].state == "present":
            rows.append((b, "contamination ruled out by test", False))
        elif lead is not None and lead.id == "H01":
            rows.append((b, "contamination suspected (process evidence)", True))
        elif h01.tier == "possible":
            rows.append((b, "contamination possible", None))
        elif lead is None or lead.id == "H05":
            rows.append((b, "unresolved", None))
        else:
            rows.append((b, f"other explanation: {lead.name}", False))
    return pd.DataFrame(rows, columns=["batch_id", "status", "is_case"])


# --------------------------------------------------------------------------- tracing


@dataclass
class CandidateSource:
    resource: str
    resource_type: str
    tier: str
    cases: list[str]
    non_cases: list[str]
    uncertain: list[str]
    cases_elsewhere: int
    non_cases_elsewhere: int
    attack_rate: float
    attack_rate_elsewhere: float
    relative_risk: float
    p_value: float
    p_adjusted: float
    first_case: str
    last_case: str
    explanation: str
    suggested_check: str

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("cases", "non_cases", "uncertain"):
            d[k] = ", ".join(d[k])
        return d


@dataclass
class TraceResult:
    graph: nx.Graph
    statuses: pd.DataFrame
    candidates: list[CandidateSource]
    clusters: list[dict]
    per_case: dict[str, list[CandidateSource]] = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([c.to_dict() for c in self.candidates])

    @property
    def leads(self) -> list[CandidateSource]:
        return [c for c in self.candidates if c.tier in ("strong lead", "possible lead")]


def _relative_risk(a: int, b: int, c: int, d: int) -> float:
    if 0 in (a + b, c + d):
        return np.nan
    if 0 in (a, b, c, d):  # Haldane-Anscombe correction
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return (a / (a + b)) / (c / (c + d))


def trace_sources(
    batches: pd.DataFrame, statuses: pd.DataFrame, kb: Knowledge | None = None
) -> TraceResult:
    kb = kb or load_knowledge()
    cfg = kb.source_tracing
    graph = genealogy_graph(batches, cfg["resources"])
    status = {b: _case(v) for b, v in statuses.set_index("batch_id")["is_case"].items()}
    starts = batches.set_index("batch_id")["start_time"].to_dict()
    cases_all = {b for b, v in status.items() if v is True}
    non_cases_all = {b for b, v in status.items() if v is False}

    raw = []
    for node, data in graph.nodes(data=True):
        if data["kind"] != "resource":
            continue
        users = set(graph.neighbors(node))
        cases = sorted(users & cases_all, key=lambda b: starts[b])
        if not cases:
            continue
        non_cases = sorted(users & non_cases_all)
        uncertain = sorted(b for b in users if status.get(b) is None)
        a, b_ = len(cases), len(non_cases)
        c, d = len(cases_all) - a, len(non_cases_all) - b_
        p = float(fisher_exact([[a, b_], [c, d]], alternative="greater")[1])
        raw.append((node, data["resource_type"], cases, non_cases, uncertain, a, b_, c, d, p))

    n_tests = max(1, sum(1 for r in raw if r[5] >= cfg["min_cases"]))
    candidates = []
    for node, rtype, cases, non_cases, uncertain, a, b_, c, d, p in raw:
        attack = a / (a + b_)
        elsewhere = c / (c + d) if c + d else np.nan
        p_adj = min(1.0, p * n_tests) if a >= cfg["min_cases"] else np.nan
        if (
            a >= cfg["min_cases"]
            and p_adj < cfg["alpha"]
            and attack >= cfg["strong_min_attack_rate"]
        ):
            tier = "strong lead"
        elif (
            a >= cfg["min_cases"]
            and _relative_risk(a, b_, c, d) >= cfg["possible_min_relative_risk"]
        ):
            tier = "possible lead"
        elif a >= cfg["min_cases"]:
            tier = "not over-represented"
        else:
            tier = "single case"
        explanation = (
            f"{a} of {a + b_} classified batches using it are contamination cases "
            f"({attack:.0%}), vs {c} of {c + d} that did not use it ({elsewhere:.0%})"
        )
        if a >= cfg["min_cases"]:
            explanation += f"; one-sided Fisher p = {p:.2g}, Bonferroni-adjusted {p_adj:.2g} over {n_tests} resources"
        if uncertain:
            explanation += f"; {len(uncertain)} user(s) with uncertain status excluded"
        explanation += ". A shared resource is a lead, not proof of cause."
        candidates.append(
            CandidateSource(
                resource=node,
                resource_type=rtype,
                tier=tier,
                cases=cases,
                non_cases=non_cases,
                uncertain=uncertain,
                cases_elsewhere=c,
                non_cases_elsewhere=d,
                attack_rate=round(attack, 3),
                attack_rate_elsewhere=round(elsewhere, 3),
                relative_risk=round(float(_relative_risk(a, b_, c, d)), 2),
                p_value=p,
                p_adjusted=p_adj,
                first_case=cases[0],
                last_case=cases[-1],
                explanation=explanation,
                suggested_check=cfg.get("resource_checks", {}).get(rtype, ""),
            )
        )

    def order(s: CandidateSource):
        return (TIER_ORDER[s.tier], np.nan_to_num(s.p_adjusted, nan=2.0), s.p_value, s.resource)

    candidates.sort(key=order)

    # Confounding check: a resource whose cases are mostly cases of a stronger lead can
    # look significant only because of that overlap. Each further strong lead must stay
    # significant once batches that used a stronger strong lead are excluded.
    kept: list[CandidateSource] = []
    for s in [c for c in candidates if c.tier == "strong lead"]:
        excluded = {b for k in kept for b in graph.neighbors(k.resource)}
        users = set(graph.neighbors(s.resource)) - excluded
        a = sum(status.get(b) is True for b in users)
        b_ = sum(status.get(b) is False for b in users)
        c = len(cases_all - excluded) - a
        d = len(non_cases_all - excluded) - b_
        p = float(fisher_exact([[a, b_], [c, d]], alternative="greater")[1]) if a else 1.0
        if not kept or (a >= cfg["min_cases"] and min(1.0, p * n_tests) < cfg["alpha"]):
            kept.append(s)
            continue
        s.tier = "possible lead"
        s.explanation = s.explanation.replace(
            ". A shared resource is a lead",
            f"; overlaps with {', '.join(k.resource for k in kept)}: only {a} case(s) remain once "
            f"those batches are excluded (Fisher p = {p:.2g}), so it may be confounded"
            ". A shared resource is a lead",
        )
    candidates.sort(key=order)

    # Clusters: cases connected through lead resources (strong or possible).
    shared = [s.resource for s in candidates if s.tier in ("strong lead", "possible lead")]
    sub = graph.subgraph(
        set(shared) | {b for s in candidates if s.resource in shared for b in s.cases}
    )
    clusters = []
    for comp in nx.connected_components(sub):
        members = sorted((n for n in comp if n in cases_all), key=lambda b: starts[b])
        via = sorted(n for n in comp if n in shared)
        if len(members) >= cfg["min_cases"]:
            clusters.append({"cases": members, "linked_by": via})

    per_case = {
        b: [s for s in candidates if b in s.cases and s.resource in shared] for b in cases_all
    }
    return TraceResult(graph, statuses, candidates, clusters, per_case)


def apply_to_reports(diagnosis: DiagnosisResult, trace: TraceResult) -> None:
    """Write candidate sources and affected scope into the investigation reports."""
    status = trace.statuses.set_index("batch_id")
    for b, report in diagnosis.reports.items():
        is_case = _case(status["is_case"].get(b))
        if is_case is not True:
            report.sources = []
            if is_case is None:
                report.assessment["affected_scope"] = "not traced (contamination status uncertain)"
            else:
                report.assessment["affected_scope"] = "not traced (contamination not suspected)"
            continue
        leads = trace.per_case.get(b, [])
        report.sources = [
            {
                "resource": s.resource,
                "tier": s.tier,
                "explanation": s.explanation,
                "suggested_check": s.suggested_check,
                "other_cases": [c for c in s.cases if c != b],
                "under_review": [
                    u for u in s.non_cases + s.uncertain if not _is_reference(status, u)
                ],
            }
            for s in leads
        ]
        if not leads:
            report.assessment["affected_scope"] = (
                "single batch (shares no input or equipment with other cases; may be an independent source)"
            )
            continue
        linked = sorted({c for s in leads for c in s.cases if c != b})
        review = sorted(
            {
                u
                for s in leads
                if s.tier == "strong lead"
                for u in s.non_cases + s.uncertain
                if not _is_reference(status, u)
            }
        )
        scope = f"linked cases {', '.join(linked)} via {', '.join(s.resource for s in leads)}"
        if review:
            scope += f"; other batches using a strong-lead resource to review: {', '.join(review)}"
        report.assessment["affected_scope"] = scope


def _is_reference(status: pd.DataFrame, batch_id: str) -> bool:
    return str(status["status"].get(batch_id, "")).startswith("approved normal")
