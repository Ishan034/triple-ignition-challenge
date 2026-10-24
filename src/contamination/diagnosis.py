"""Layer 5, diagnostic reasoning; Layer 3, confidence and severity; Layer 4, next checks.

This step goes from "something unusual happened" to "which explanations fit the
evidence, what is still unknown, and what should be checked next".

* Evidence items are computed from detection outputs (scores and alerts) and from
  any scientist-recorded investigations. Each item is ``present``, ``absent`` or
  ``unknown``.
* The hypothesis catalogue and evidence matrix live in
  ``knowledge/catalogue.yaml``, so scientists can review them without reading code.
* Hypotheses are ranked by a transparent score (supporting weights minus
  contradicting weights). The score orders explanations for review. It is not a
  probability, and the system never declares contamination from process data alone:
  only an independent test can move contamination evidence to "confirmed".
* Confidence (how well supported the leading explanation is) and severity
  dimensions (deviation, quality impact, scope) are reported separately.

Everything is computed as of a point in time: with ``as_of_h`` only results available
by then are used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .alerts import Alert
from .monitor import DEFAULT_CONFIG, MonitorConfig, MonitorResult, monitor
from .reference import ReferenceProfile
from .schema import INVESTIGATION_COLUMNS

CATALOGUE_PATH = Path(__file__).parent / "knowledge" / "catalogue.yaml"
# Process roles the evidence rules refer to; each catalogue maps them to its own
# variable names (e.g. biomass -> biomass_od600 for the alpha-amylase proxy).
ROLES = ("biomass", "oxygen", "ph", "product", "temperature")
# Controller outputs (e.g. base pump, cooling-water valve). Optional: a process that does
# not record them maps them to "not_measured" and the evidence that needs them is unknown.
OPTIONAL_ROLES = ("ph_control", "temperature_control")
NOT_MEASURED = "not_measured"
COST_RANK = {"low": 0, "medium": 1, "high": 2}
CAVEAT = (
    "Decision support only. Rankings order explanations for scientist review. They are "
    "not probabilities, and no organism is identified from process data. The knowledge "
    "base is a draft that has not been validated by MaverickX scientists."
)


@dataclass(frozen=True)
class Knowledge:
    review_status: str
    parameters: dict
    evidence: dict
    hypotheses: dict
    checks: dict
    always_consider: dict = field(default_factory=dict)
    source_tracing: dict = field(default_factory=dict)
    roles: dict = field(default_factory=dict)
    control_loops: dict = field(default_factory=dict)
    # Physically linked signals for the linked-shift detector (measured variable names).
    linked_signals: tuple[str, ...] = ()

    @property
    def linked(self) -> tuple[str, ...]:
        """Variables expected to move together in a real culture change."""
        return (self.roles["ph"], self.roles["oxygen"], self.roles["biomass"])

    @property
    def measured_control_loops(self) -> dict[str, tuple[str, bool]]:
        """{controlled variable: (controller output variable, output raises variable)}
        for the loops whose controller output this process records."""
        out = {}
        for controlled, loop in self.control_loops.items():
            effort = self.roles.get(loop["effort"], NOT_MEASURED)
            if effort != NOT_MEASURED and self.roles.get(controlled, NOT_MEASURED) != NOT_MEASURED:
                out[self.roles[controlled]] = (effort, bool(loop["output_raises_variable"]))
        return out

    @property
    def control_variables(self) -> set[str]:
        """Measured controller outputs: they show how hard a loop works, not the culture."""
        return {
            self.roles[r] for r in OPTIONAL_ROLES if self.roles.get(r, NOT_MEASURED) != NOT_MEASURED
        }


PROFILES_DIR = CATALOGUE_PATH.parent / "profiles"


def load_knowledge(
    path: str | Path = CATALOGUE_PATH, profile: str | Path | None = None
) -> Knowledge:
    """Load the rule book, optionally adapted to another process by a profile.

    A profile is a small YAML file (``knowledge/profiles/<name>.yaml``, or a path) with
    ``roles`` (replacing the role -> variable map) and ``parameters`` (overriding single
    parameters). Hypotheses, evidence links and checks are shared across processes. A
    campaign's ``profile.yaml`` may instead say ``use: <name>`` to follow a packaged
    profile, so it stays current when the packaged one changes.
    """
    d = yaml.safe_load(Path(path).read_text())
    if profile is not None:
        ppath = Path(profile)
        if not ppath.exists():
            ppath = PROFILES_DIR / f"{profile}.yaml"
        prof = yaml.safe_load(ppath.read_text())
        if set(prof) == {"use"}:
            ppath = PROFILES_DIR / f"{prof['use']}.yaml"
            prof = yaml.safe_load(ppath.read_text())
        d["roles"] = prof.get("roles", d.get("roles", {}))
        if "linked_signals" in prof:
            d["linked_signals"] = prof["linked_signals"]
        d["parameters"] = {**d["parameters"], **prof.get("parameters", {})}
        unknown = set(prof.get("parameters", {})) - set(
            yaml.safe_load(Path(path).read_text())["parameters"]
        )
        if unknown:
            raise ValueError(f"profile {ppath.name} sets unknown parameters {sorted(unknown)}")
        d["review_status"] = (
            f"{d['review_status']}; process profile: {prof.get('name', ppath.stem)}"
        )
    known = set(d["evidence"])
    for hid, h in d["hypotheses"].items():
        unknown = (
            set(h.get("supports", {}))
            | set(h.get("contradicts", {}))
            | set(h.get("leading_requires_any", []))
        ) - known
        if unknown:
            raise ValueError(f"{hid} links to undefined evidence {sorted(unknown)}")
    for cid, c in d["checks"].items():
        if set(c["discriminates"]) - set(d["hypotheses"]):
            raise ValueError(f"{cid} references an undefined hypothesis")
        for res in ("positive", "negative"):
            if res in c and c[res] not in known:
                raise ValueError(f"{cid} {res} result maps to undefined evidence {c[res]}")
    roles = {**{r: NOT_MEASURED for r in OPTIONAL_ROLES}, **d.get("roles", {})}
    missing = set(ROLES) - set(roles)
    if missing:
        raise ValueError(f"catalogue roles missing: {sorted(missing)}")
    loops = d.get("control_loops", {})
    for controlled, loop in loops.items():
        if controlled not in ROLES or loop.get("effort") not in OPTIONAL_ROLES:
            raise ValueError(
                f"control loop {controlled} -> {loop.get('effort')} uses unknown roles"
            )
    return Knowledge(
        d["review_status"],
        d["parameters"],
        d["evidence"],
        d["hypotheses"],
        d["checks"],
        d.get("always_consider", {}),
        d.get("source_tracing", {}),
        roles,
        loops,
        tuple(roles.get(x, x) for x in d.get("linked_signals", [])),
    )


@dataclass
class EvidenceItem:
    id: str
    name: str
    state: str  # present | absent | unknown
    detail: str
    independent: bool = False


@dataclass
class HypothesisAssessment:
    id: str
    name: str
    score: int
    tier: str  # leading | possible | not supported | fallback
    supporting: list[str] = field(default_factory=list)
    contradicting: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


@dataclass
class DiagnosisReport:
    batch_id: str
    as_of_h: float | None
    reference_version: str
    review_status: str
    alerts: list[Alert]
    evidence: dict[str, EvidenceItem]
    hypotheses: list[HypothesisAssessment]
    assessment: dict
    next_checks: list[dict]
    caveat: str = CAVEAT
    sources: list[dict] = field(default_factory=list)  # filled by tracing.apply_to_reports

    @property
    def leading(self) -> HypothesisAssessment | None:
        return next((h for h in self.hypotheses if h.tier == "leading"), None)

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "as_of_h": self.as_of_h,
            "reference_version": self.reference_version,
            "review_status": self.review_status,
            "evidence": {k: asdict(v) for k, v in self.evidence.items()},
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "assessment": self.assessment,
            "next_checks": self.next_checks,
            "sources": self.sources,
            "caveat": self.caveat,
        }

    def to_markdown(self) -> str:
        ev = self.evidence
        when = f"as of {self.as_of_h:g} h" if self.as_of_h is not None else "full record"
        lines = [
            f"# Batch investigation report: {self.batch_id}",
            f"*{when} · reference {self.reference_version} · knowledge base: {self.review_status}*",
            "",
            "## 1. What changed?",
        ]
        by_var: dict[str, list[Alert]] = {}
        linked = [a for a in self.alerts if a.detector == "linked_shift"]
        for a in self.alerts:
            if a.detector != "linked_shift":
                by_var.setdefault(a.variable, []).append(a)
        if linked:
            first = min(linked, key=lambda a: a.first_raised_h)
            lines.append(
                f"- **Linked signals changed together** from {first.start_h:g} h "
                f"(first raised {first.first_raised_h:g} h): {first.detail.split(': ', 1)[-1]}"
            )
        if not by_var and not linked:
            lines.append("- No alerts.")
        for var, al in sorted(by_var.items(), key=lambda kv: min(a.first_raised_h for a in kv[1])):
            first = min(al, key=lambda a: a.first_raised_h)
            peak = max(al, key=lambda a: abs(a.robust_z_at_peak))
            persist = "persistent" if any(a.persistence == "persistent" for a in al) else "isolated"
            lines.append(
                f"- **{var}**: {peak.direction} from {min(a.start_h for a in al):g} h "
                f"(first raised {first.first_raised_h:g} h, {persist}); peak observed "
                f"{peak.observed_at_peak:.3g} vs reference {peak.reference_median_at_peak:.3g} "
                f"(robust z {peak.robust_z_at_peak:+.1f}); detectors: "
                f"{', '.join(sorted({a.detector for a in al}))}"
            )
        lines += ["", "## 2. Plausible explanations (ranked for review, not probabilities)"]
        for h in self.hypotheses:
            sup = ", ".join(f"{e} {ev[e].name}" for e in h.supporting) or "none"
            con = ", ".join(f"{e} {ev[e].name}" for e in h.contradicting) or "none"
            lines.append(f"- **{h.name}** ({h.id}): {h.tier}, score {h.score:+d}")
            if h.id != "H05":
                lines.append(f"  - supporting: {sup}")
                lines.append(f"  - contradicting: {con}")
        lines += ["", "## 3. What information is missing?"]
        missing = sorted({m for h in self.hypotheses[:2] for m in h.missing})
        lines += [f"- {m} {ev[m].name}: {ev[m].detail}" for m in missing] or [
            "- Nothing flagged as missing."
        ]
        lines += ["", "## 4. Assessment (dimensions kept separate)"]
        lines += [f"- **{k.replace('_', ' ')}**: {v}" for k, v in self.assessment.items()]
        lines += ["", "## 5. Possible sources (leads for investigation, not proof)"]
        if not self.sources:
            lines.append("- Source tracing not run, or contamination not suspected for this batch.")
        for src in self.sources:
            lines.append(f"- **{src['resource']}** ({src['tier']}): {src['explanation']}")
            if src.get("suggested_check"):
                lines.append(f"  - suggested check: {src['suggested_check']}")
        lines += ["", "## 6. Recommended next checks"]
        lines += [
            f"{i}. {c['name']} (~{c['turnaround_h']:g} h, {c['relative_cost']} cost): {c['reason']}"
            for i, c in enumerate(self.next_checks, 1)
        ] or ["- No further checks suggested."]
        lines += [
            "",
            "## 7. Scientist decision",
            "- Reviewed by: ____  ·  Checks performed: ____  ·  Final assessment: ____",
            "",
            "## Evidence detail",
        ]
        lines += [f"- {e.id} {e.name}: **{e.state}**. {e.detail}" for e in ev.values()]
        lines += ["", f"> {self.caveat}"]
        return "\n".join(lines)


@dataclass
class DiagnosisResult:
    monitor_result: MonitorResult
    reports: dict[str, DiagnosisReport]

    def table(self) -> pd.DataFrame:
        rows = []
        for r in self.reports.values():
            lead = r.leading
            rows.append(
                {
                    "batch_id": r.batch_id,
                    "leading_id": lead.id if lead else "",
                    "ranking": " > ".join(f"{h.id}({h.score:+d})" for h in r.hypotheses),
                    **r.assessment,
                    "next_check": r.next_checks[0]["name"] if r.next_checks else "",
                }
            )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- evidence


def _var(s: pd.DataFrame, v: str) -> pd.DataFrame:
    return s[s["variable"] == v].sort_values("elapsed_h")


def _nearest_z(series: pd.DataFrame, t: float, tol: float) -> float:
    """z of the most recent reading at or before t (within tol hours)."""
    prior = series[(series["elapsed_h"] <= t) & (series["elapsed_h"] >= t - tol)]
    return float(prior["z"].iloc[-1]) if len(prior) else np.nan


def _shifted_z(
    df: pd.DataFrame, reference: ReferenceProfile, variable: str, shift: float
) -> pd.DataFrame:
    """Re-score readings against the reference shifted ``shift`` hours later in time."""
    ref = reference.stats[reference.stats["variable"] == variable].sort_values("hour")
    if not len(df) or not len(ref) or shift == 0:
        return df
    tt = df["elapsed_h"].to_numpy() - shift
    hours = ref["hour"].to_numpy()
    med = np.interp(tt, hours, ref["median"].to_numpy(), left=np.nan, right=np.nan)
    scale = np.interp(tt, hours, ref["scale"].to_numpy(), left=np.nan, right=np.nan)
    out = df.copy()
    out["ref_median"], out["z"] = med, (df["value"].to_numpy() - med) / scale
    return out


def _delayed_trajectory(
    od: pd.DataFrame, reference: ReferenceProfile, p: dict, biomass_var: str
) -> tuple[str, str, float]:
    ref = reference.stats[reference.stats["variable"] == biomass_var].sort_values("hour")
    if ref.empty:
        return "unknown", f"No reference for {biomass_var}: biomass is not monitored here.", 0.0
    hours, med, scale = ref["hour"].to_numpy(), ref["median"].to_numpy(), ref["scale"].to_numpy()
    # Fit timing on the growth phase only; the stationary phase says little about delay.
    lo_f, hi_f = p["growth_phase_fraction"]
    final = float(med.max())
    od = od[od["z"].notna() & od["value"].between(lo_f * final, hi_f * final)]
    if len(od) < p["shift_min_points"]:
        return (
            "unknown",
            "Not enough growth-phase biomass readings yet to compare trajectory timing.",
            0.0,
        )
    t, x = od["elapsed_h"].to_numpy(), od["value"].to_numpy()

    def residual(shift: float) -> float:
        tt = t - shift
        ok = (tt >= hours.min()) & (tt <= hours.max())
        if ok.sum() < p["shift_min_points"]:
            return np.inf
        m, sc = np.interp(tt[ok], hours, med), np.interp(tt[ok], hours, scale)
        return float(np.median(np.abs(x[ok] - m) / sc))

    lo, hi = p["shift_search_h"]
    shifts = np.arange(lo, hi + 0.5, 0.5)
    res = np.array([residual(s) for s in shifts])
    best = int(np.argmin(res))
    r0 = residual(0.0)
    detail = (
        f"Growth phase best fits the reference shifted {shifts[best]:+g} h "
        f"(median |z| {res[best]:.2f} vs {r0:.2f} unshifted, {len(od)} readings)."
    )
    present = (
        shifts[best] >= p["shift_min_h"]
        and res[best] <= p["shift_max_residual"]
        and r0 - res[best] >= p["shift_min_improvement"]
    )
    return ("present" if present else "absent"), detail, float(shifts[best]) if present else 0.0


def _decoupled(
    s: pd.DataFrame, alerts: list[Alert], p: dict, linked: tuple[str, ...]
) -> tuple[str, str]:
    found, too_early = [], []
    for var in linked:
        windows = [(a.start_h, a.end_h) for a in alerts if a.variable == var]
        if not windows:
            continue
        duration = max(b for _, b in windows) - min(a for a, _ in windows)
        ok_all = True
        for other in linked:
            if other == var:
                continue
            o = _var(s, other)
            o = o[o["z"].notna()]
            in_win = (
                o[np.any([o["elapsed_h"].between(a - 1, b + 1) for a, b in windows], axis=0)]
                if len(o)
                else o
            )
            if not len(in_win):
                ok_all = False
                break
            frac = float((in_win["z"].abs() < p["linked_in_range_z"]).mean())
            if frac < p["decoupled_min_fraction"]:
                ok_all = False
                break
        if ok_all:
            (found if duration >= p["decoupled_min_duration_h"] else too_early).append(var)
    if found:
        return (
            "present",
            f"{', '.join(found)} deviated while linked signals stayed within reference.",
        )
    if too_early:
        return (
            "unknown",
            (
                f"{', '.join(too_early)} deviates alone so far, but for under "
                f"{p['decoupled_min_duration_h']:g} h: too early to tell whether linked signals "
                "will follow."
            ),
        )
    return "absent", "Deviations are accompanied by changes in linked signals."


def _control_loop_evidence(
    s: pd.DataFrame, alerts: list[Alert], p: dict, kb: Knowledge
) -> tuple[tuple[str, str], tuple[str, str]]:
    """E17 (a controller works unusually hard while holding its variable in range) and
    E18 (a controlled variable drifts while its controller output does not respond).

    A healthy loop answers a drift in its variable by pushing its output in the
    correcting direction. If the variable drifts and the output stays at or below its
    normal level, the actuator (pump, valve, cooling supply) is the first suspect.
    """
    r, dz = kb.roles, p["deviation_z"]
    measured = [
        (controlled, loop)
        for controlled, loop in kb.control_loops.items()
        if r.get(loop["effort"], NOT_MEASURED) != NOT_MEASURED
        and len(_var(s, r[loop["effort"]]))
        and len(_var(s, r[controlled]))
    ]
    if not measured:
        why = "No controller outputs are recorded for this process."
        return ("unknown", why), ("unknown", why)

    working, unresponsive, notes = [], [], []
    for controlled, loop in measured:
        cvar, evar = r[controlled], r[loop["effort"]]
        effort = _var(s, evar)
        effort = effort[effort["z"].notna()]
        level = _var(s, cvar)
        level = level[level["z"].notna()]
        # E18: the controlled variable drifts persistently; which way should the output move?
        for a in alerts:
            if a.variable != cvar or a.persistence != "persistent":
                continue
            raise_needed = a.direction == "low"
            needed = "high" if raise_needed == bool(loop["output_raises_variable"]) else "low"
            # Judge the response where the drift begins: later in a long episode the
            # fault may have cleared and the controller be catching up.
            win = effort[
                effort["elapsed_h"].between(a.start_h - 2, min(a.end_h, a.start_h + 3) + 0.5)
            ]
            if len(win) < 2:
                continue
            push = win["z"] if needed == "high" else -win["z"]
            if float(push.median()) < p["control_response_z"]:
                unresponsive.append(
                    f"{cvar} {a.direction} from {a.start_h:g} h while {evar} did not push back "
                    f"(median {float(push.median()):+.1f} z in the correcting direction)"
                )
        # E17: the output deviates persistently while the controlled variable stays in range.
        for a in alerts:
            if a.variable != evar or a.persistence != "persistent":
                continue
            win = level[level["elapsed_h"].between(a.start_h - 1, a.end_h + 1)]
            if len(win) and float((win["z"].abs() < dz).mean()) >= p["decoupled_min_fraction"]:
                working.append(
                    f"{evar} {a.direction} {a.start_h:g}-{a.end_h:g} h while {cvar} stayed in range"
                )
        notes.append(f"{controlled} loop: {cvar} controlled by {evar}")
    e17 = (
        ("present", "; ".join(working) + ".")
        if working
        else ("absent", f"No unusual controller effort ({'; '.join(notes)}).")
    )
    e18 = (
        ("present", "; ".join(unresponsive) + ".")
        if unresponsive
        else ("absent", f"Controllers responded normally to any drift ({'; '.join(notes)}).")
    )
    return e17, e18


def _investigation_evidence(investigations: pd.DataFrame, batch_id: str, kb: Knowledge) -> dict:
    """Evidence from scientist-recorded check results, mapped through the catalogue.

    An item is ``unknown`` until a check that can produce it has been recorded, then
    ``present`` (a matching result) or ``absent``.
    """
    inv = (
        investigations[investigations["batch_id"] == batch_id]
        if len(investigations)
        else investigations
    )
    out = {}
    for eid in kb.evidence:
        family = {
            cid: res
            for cid, c in kb.checks.items()
            for res in ("positive", "negative")
            if c.get(res) == eid
        }
        if not family:
            continue
        done = inv[inv["check_id"].isin(family)]
        hits = (
            done[[family[r.check_id] == r.result for r in done.itertuples()]] if len(done) else done
        )
        if len(hits):
            detail = ", ".join(
                f"{r.check_id} {r.result} at {r.recorded_h:g} h"
                + (f" ({r.note})" if isinstance(getattr(r, "note", None), str) else "")
                for r in hits.itertuples()
            )
            out[eid] = ("present", detail + ".")
        elif len(done):
            out[eid] = (
                "absent",
                f"Checked ({', '.join(sorted(set(done['check_id'])))}); result did not indicate this.",
            )
        else:
            out[eid] = ("unknown", f"Not checked yet ({', '.join(sorted(family))} would tell).")
    return out


def compute_evidence(
    batch_id: str,
    scores: pd.DataFrame,
    alerts: list[Alert],
    measurements: pd.DataFrame,
    investigations: pd.DataFrame,
    reference: ReferenceProfile,
    kb: Knowledge,
) -> dict[str, EvidenceItem]:
    p = kb.parameters
    r = kb.roles
    dz = p["deviation_z"]
    s = scores[scores["batch_id"] == batch_id]
    control = kb.control_variables
    # A linked-shift alert sits on the group's first signal but stands for several
    # signals changing together: it is evidence E01, not a deviation of that one signal.
    linked = [a for a in alerts if a.detector == "linked_shift"]
    alerts = [a for a in alerts if a.detector != "linked_shift"]
    persistent = {a.variable for a in alerts if a.persistence == "persistent"} - control
    alerted = {a.variable for a in alerts} - control
    has = {r[role]: bool(len(_var(s, r[role]))) for role in ROLES}
    out: dict[str, tuple[str, str]] = {}

    multi = sorted(persistent - {r["temperature"]})
    out["E01"] = (
        "present" if len(multi) >= p["multi_signal_min_variables"] else "absent",
        f"Persistent alerts on: {', '.join(multi) or 'none'}.",
    )
    persistent_linked = [a for a in linked if a.persistence == "persistent"]
    if persistent_linked:
        first = min(persistent_linked, key=lambda a: a.start_h)
        out["E01"] = (
            "present",
            f"Linked signals changed together from {first.start_h:g} h ({first.detail}).",
        )
    out["E02"] = (
        "present" if len(alerted) == 1 else "absent",
        f"Variables alerted: {', '.join(sorted(alerted)) or 'none'}.",
    )

    thr = [a for a in alerts if a.detector == "robust_threshold"]
    if thr and all(a.persistence == "isolated" for a in thr):
        back = all(abs(_var(s, v)["z"].dropna().iloc[-1]) < dz for v in {a.variable for a in thr})
        out["E03"] = (
            "present" if back else "absent",
            f"{len(thr)} isolated threshold alert(s); latest readings {'back within' if back else 'still outside'} reference.",
        )
    else:
        out["E03"] = (
            "absent",
            "Threshold alerts are not limited to isolated readings."
            if thr
            else "No threshold alerts.",
        )

    *out["E04"], delay_h = _delayed_trajectory(_var(s, r["biomass"]), reference, p, r["biomass"])
    out["E04"] = tuple(out["E04"])

    def persistent_dir(var: str, direction: str, eid: str, what: str) -> None:
        if not has[var]:
            out[eid] = ("unknown", f"No {var} readings available.")
            return
        hit = [
            a
            for a in alerts
            if a.variable == var and a.persistence == "persistent" and a.direction == direction
        ]
        out[eid] = (
            "present" if hit else "absent",
            f"{what} from {min(a.start_h for a in hit):g} h."
            if hit
            else f"No persistent {direction} alert on {var}.",
        )

    persistent_dir(r["biomass"], "high", "E05", "Biomass above reference")

    # If the whole trajectory is delayed, judge activity and biomass against the
    # time-shifted reference, so late product formation is not mistaken for low activity.
    act = _shifted_z(_var(s, r["product"]), reference, r["product"], delay_h)
    act = act[act["ref_median"] >= p["product_phase_min_activity"]]
    od = _shifted_z(_var(s, r["biomass"]), reference, r["biomass"], delay_h)
    if not len(act):
        out["E06"] = ("unknown", "No enzyme-activity results from the product-forming phase yet.")
    else:
        low = [
            (r.elapsed_h, r.z)
            for r in act.itertuples()
            if r.z <= -dz and _nearest_z(od, r.elapsed_h, 2.0) >= -1.0
        ]
        out["E06"] = (
            "present" if low else "absent",
            f"Activity low with normal/high biomass at {', '.join(f'{h:g} h' for h, _ in low)}."
            if low
            else f"{len(act)} product-phase activity result(s); none low relative to biomass.",
        )
        if delay_h:
            out["E06"] = (
                out["E06"][0],
                out["E06"][1] + f" (compared with reference shifted {delay_h:+g} h)",
            )

    persistent_dir(r["ph"], "low", "E07", "pH below reference")

    do = _var(s, r["oxygen"])
    if not len(do) or not len(od):
        out["E08"] = ("unknown", "Needs both DO and OD600 readings.")
    else:
        hits = [
            r.elapsed_h
            for r in do.itertuples()
            if r.z <= -dz and _nearest_z(od, r.elapsed_h, 2.0) >= dz
        ]
        out["E08"] = (
            "present" if len(hits) >= 2 else "absent",
            f"DO low while OD600 high at {len(hits)} reading(s).",
        )

    out["E09"] = _decoupled(s, alerts, p, kb.linked)

    m = measurements[measurements["batch_id"] == batch_id]
    notes = m[m["observation"].notna() & (m["observation"].astype(str).str.strip() != "")]
    out["E10"] = (
        ("present", f'{notes.iloc[0]["elapsed_h"]:g} h: "{notes.iloc[0]["observation"]}"')
        if len(notes)
        else ("absent", "No observations recorded.")
    )

    dq = [w for a in alerts for w in a.data_quality_warnings]
    out["E11"] = (
        "present" if dq else "absent",
        "; ".join(sorted(set(dq))) or "No data-quality warnings near alerts.",
    )

    out.update(_investigation_evidence(investigations, batch_id, kb))

    persistent_dir(r["temperature"], "high", "E14", "Temperature above reference")
    if out["E14"][0] == "absent" and has[r["temperature"]]:
        persistent_dir(r["temperature"], "low", "E14", "Temperature below reference")

    if "E17" in kb.evidence:
        out["E17"], out["E18"] = _control_loop_evidence(s, alerts, p, kb)

    return {
        eid: EvidenceItem(
            eid, kb.evidence[eid]["name"], state, detail, bool(kb.evidence[eid].get("independent"))
        )
        for eid, (state, detail) in sorted(out.items())
    }


# --------------------------------------------------------------------------- reasoning


def assess(evidence: dict[str, EvidenceItem], kb: Knowledge) -> list[HypothesisAssessment]:
    ranked = []
    for hid, h in kb.hypotheses.items():
        if hid == "H05":
            continue
        sup = {e: w for e, w in h.get("supports", {}).items() if evidence[e].state == "present"}
        con = {e: w for e, w in h.get("contradicts", {}).items() if evidence[e].state == "present"}
        linked = set(h.get("supports", {})) | set(h.get("contradicts", {}))
        missing = sorted(e for e in linked if evidence[e].state == "unknown")
        ranked.append(
            HypothesisAssessment(
                hid,
                h["name"],
                sum(sup.values()) - sum(con.values()),
                "",
                sorted(sup),
                sorted(con),
                missing,
            )
        )
    ranked.sort(key=lambda a: (-a.score, a.id))
    top = ranked[0].score
    second = ranked[1].score if len(ranked) > 1 else -np.inf
    for i, a in enumerate(ranked):
        if a.score <= 0:
            a.tier = "not supported"
        elif (
            i == 0
            and top - second >= 1
            and a.score >= kb.parameters["leading_min_score"]
            and _has_specific(a, kb)
        ):
            a.tier = "leading"
        else:
            a.tier = "possible"
    unresolved = HypothesisAssessment(
        "H05",
        kb.hypotheses["H05"]["name"],
        0,
        "fallback" if ranked[0].tier == "leading" else "leading",
    )
    return ranked + [unresolved]


def _has_specific(a: HypothesisAssessment, kb: Knowledge) -> bool:
    required = kb.hypotheses[a.id].get("leading_requires_any")
    return not required or bool(set(required) & set(a.supporting))


def _confidence(hyps: list[HypothesisAssessment], evidence: dict[str, EvidenceItem]) -> str:
    lead = next((h for h in hyps if h.tier == "leading"), None)
    if lead is None or lead.id == "H05":
        return "limited (no single explanation is supported)"
    if any(evidence[e].independent for e in lead.supporting):
        return "strong (supported by an independent test)"
    others = [h.score for h in hyps if h.id not in (lead.id, "H05")]
    margin = lead.score - max(others) if others else lead.score
    if margin >= 2 and len(lead.supporting) >= 2 and not lead.contradicting:
        return "moderate (consistent process evidence, no independent test)"
    return "limited (process evidence only, or competing explanations)"


def _next_checks(
    hyps: list[HypothesisAssessment], investigations: pd.DataFrame, batch_id: str, kb: Knowledge
) -> list[dict]:
    done = (
        set(investigations.loc[investigations["batch_id"] == batch_id, "check_id"])
        if len(investigations)
        else set()
    )
    candidates = [h for h in hyps if h.id != "H05"]
    focus = [h.id for h in candidates if h.score > 0][:2]
    for h in candidates:  # always compare at least two explanations
        if len(focus) >= 2:
            break
        if h.id not in focus:
            focus.append(h.id)
    by_id = {h.id: h for h in hyps}
    for hid, rule in kb.always_consider.items():
        h = by_id.get(hid)
        ruled_out = (
            rule.get("ruled_out_by") and h is not None and rule["ruled_out_by"] in h.contradicting
        )
        if (
            h is not None
            and h.score >= rule.get("min_score", 0)
            and not ruled_out
            and hid not in focus
        ):
            focus.append(hid)
    confirmed = {h.id for h in hyps if any(kb.evidence[e].get("independent") for e in h.supporting)}
    names = {h.id: h.name for h in hyps}
    options = []
    for cid, c in kb.checks.items():
        covers = [h for h in focus if h in c["discriminates"]]
        if c.get("screen"):
            covers = [h for h in covers if h not in confirmed]
        if cid in done or not covers:
            continue
        options.append(
            (
                (-len(covers), COST_RANK.get(c["relative_cost"], 9), c["turnaround_h"]),
                {
                    "check_id": cid,
                    "name": c["name"],
                    "turnaround_h": float(c["turnaround_h"]),
                    "relative_cost": c["relative_cost"],
                    "reason": "informative about "
                    + " and ".join(f"{names[h]} ({h})" for h in covers),
                },
            )
        )
    return [o for _, o in sorted(options, key=lambda x: x[0])[:3]]


def _assessment(batch_id, hyps, evidence, alerts, summary_row, scores, kb) -> dict:
    if evidence["E12"].state == "present":
        cont = "confirmed by independent test"
    elif evidence["E13"].state == "present":
        cont = "independent test negative"
    else:
        cont = "unconfirmed (no contamination test recorded)"
    product = kb.roles["product"]
    act = scores[(scores["batch_id"] == batch_id) & (scores["variable"] == product)]
    act = act[act["ref_median"] >= kb.parameters["product_phase_min_activity"]]
    low_act = [
        a
        for a in alerts
        if a.variable == product and a.direction == "low" and a.persistence == "persistent"
    ]
    if low_act:
        quality = f"deviation established ({product} below reference from {min(a.start_h for a in low_act):g} h)"
    elif len(act):
        quality = "no deviation established"
    else:
        quality = "not assessed yet (no product-phase activity results)"
    lead = next((h for h in hyps if h.tier == "leading"), None)
    return {
        "leading_explanation": lead.name if lead else "no single leading explanation",
        "contamination_evidence": cont,
        "process_deviation": summary_row["process_deviation"],
        "quality_impact": quality,
        "affected_scope": "single batch (source tracing not yet run)",
        "confidence": _confidence(hyps, evidence),
    }


def diagnose(
    measurements: pd.DataFrame,
    reference: ReferenceProfile,
    config: MonitorConfig = DEFAULT_CONFIG,
    investigations: pd.DataFrame | None = None,
    as_of_h: float | None = None,
    kb: Knowledge | None = None,
) -> DiagnosisResult:
    """Run monitoring, then build an investigation report for every batch with alerts."""
    kb = kb or load_knowledge()
    inv = (
        investigations
        if investigations is not None
        else pd.DataFrame(columns=INVESTIGATION_COLUMNS)
    )
    if as_of_h is not None:
        measurements = measurements[measurements["available_h"] <= as_of_h]
        inv = inv[inv["recorded_h"] <= as_of_h]
    result = monitor(measurements, reference, config)
    summary = result.summary.set_index("batch_id")

    reports = {}
    present = set(measurements["batch_id"])
    batches = sorted({a.batch_id for a in result.alerts} | (set(inv["batch_id"]) & present))
    for b in batches:
        alerts = [a for a in result.alerts if a.batch_id == b]
        evidence = compute_evidence(b, result.scores, alerts, measurements, inv, reference, kb)
        hyps = assess(evidence, kb)
        reports[b] = DiagnosisReport(
            batch_id=b,
            as_of_h=as_of_h,
            reference_version=reference.version,
            review_status=kb.review_status,
            alerts=alerts,
            evidence=evidence,
            hypotheses=hyps,
            assessment=_assessment(b, hyps, evidence, alerts, summary.loc[b], result.scores, kb),
            next_checks=_next_checks(hyps, inv, b, kb),
        )
    return DiagnosisResult(result, reports)
