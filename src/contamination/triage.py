"""What needs attention first: a priority per batch, batches at risk through a shared
source, and a plain-text summary to pass on when escalating.

Priorities order the review queue; they are not probabilities, and contamination
evidence is kept apart from how abnormal a batch is:

* ``act now``: contamination suspected from process evidence, or confirmed by a test.
* ``review``: abnormal, with another leading explanation or contamination ruled out.
* ``at risk``: no alert, but the batch used a resource that source tracing ranks as a
  lead (a strong or possible lead), so it shares a possible source with the cases.
* ``no action``: within the reference profile and not linked to a lead.
"""

from __future__ import annotations

import pandas as pd

PRIORITIES = ("act now", "review", "at risk", "no action")
LEAD_TIERS = ("strong lead", "possible lead")


def at_risk(trace, batch_ids=None, tiers=LEAD_TIERS) -> dict[str, list[str]]:
    """Batches that used a lead resource without being a case: {batch: [resource (tier)]}."""
    out: dict[str, list[str]] = {}
    for c in trace.candidates:
        if c.tier not in tiers or c.resource not in trace.graph:
            continue
        for b in trace.graph.neighbors(c.resource):
            if b in c.cases or (batch_ids is not None and b not in batch_ids):
                continue
            out.setdefault(b, []).append(f"{c.resource} ({c.tier})")
    return out


def _priority(row, risk: list[str]) -> tuple[str, str]:
    status = str(row["status"])
    if status.startswith(("contamination suspected", "contamination confirmed")):
        return "act now", status
    if row["process_deviation"] not in ("within reference", "no data"):
        lead = row["leading_explanation"] or "unresolved"
        return "review", f"{row['process_deviation']}; leading: {lead}"
    if risk:
        return "at risk", "used " + ", ".join(risk)
    return "no action", "within the reference profile"


def triage(overview: pd.DataFrame, trace, actions: pd.DataFrame | None = None) -> pd.DataFrame:
    """The overview table with ``priority``, ``why`` and ``open_actions``, most urgent first."""
    risk = at_risk(trace, set(overview["batch_id"]))
    df = overview.copy()
    pr = [_priority(r, risk.get(r["batch_id"], [])) for _, r in df.iterrows()]
    df.insert(1, "priority", [p for p, _ in pr])
    df.insert(2, "why", [w for _, w in pr])
    open_actions = (
        actions[actions["status"] == "open"].groupby("batch_id").size()
        if actions is not None and len(actions)
        else pd.Series(dtype=int)
    )
    df["open_actions"] = df["batch_id"].map(open_actions).fillna(0).astype(int)
    order = {p: i for i, p in enumerate(PRIORITIES)}
    df["_o"] = df["priority"].map(order)
    df = df.sort_values(["_o", "first_alert_h"], na_position="last", kind="stable")
    return df.drop(columns="_o").reset_index(drop=True)


def headline(t: pd.DataFrame) -> tuple[int, str]:
    """(number of batches needing attention, one-line breakdown)."""
    counts = t["priority"].value_counts()
    parts = [f"{counts[p]} {p}" for p in PRIORITIES[:3] if counts.get(p, 0)]
    n = int(sum(counts.get(p, 0) for p in PRIORITIES[:3]))
    return n, ", ".join(parts)


def escalation_text(batch_id: str, report, plant_name: str = "", as_of_h=None) -> str:
    """A short plain-text summary to paste into an email or message when escalating.

    Nothing is sent from here: the user copies or downloads it."""
    a = report.assessment
    when = f" (as of {as_of_h:g} h)" if as_of_h is not None else ""
    lines = [
        f"Batch {batch_id}{when}: needs review" + (f" ({plant_name})" if plant_name else ""),
        "",
        f"Leading explanation: {a['leading_explanation']}",
        f"Contamination evidence: {a['contamination_evidence']}",
        f"Process deviation: {a['process_deviation']}",
        f"Confidence: {a['confidence']}",
        f"Quality impact: {a['quality_impact']}",
        f"Affected scope: {a['affected_scope']}",
        "",
        "What changed:",
    ]
    for al in sorted(report.alerts, key=lambda x: x.first_raised_h)[:5]:
        lines.append(
            f"- {al.variable} {al.direction} from {al.start_h:g} h: "
            f"{al.observed_at_peak:.3g} vs reference {al.reference_median_at_peak:.3g} "
            f"({al.detector})"
        )
    if report.next_checks:
        ch = report.next_checks[0]
        lines += [
            "",
            f"Suggested next check: {ch['name']} (~{ch['turnaround_h']:g} h): {ch['reason']}",
        ]
    if report.sources:
        lines += ["", "Possible sources (leads, not proof):"]
        lines += [f"- {s['resource']} ({s['tier']})" for s in report.sources[:3]]
    note = (
        "Decision support only: rankings are not probabilities, and no organism is "
        "identified from process data."
    )
    lines += ["", note]
    return "\n".join(lines)
