"""Contamination investigation dashboard (Streamlit).

    streamlit run app/dashboard.py

Pages: campaign overview, batch investigation (with time replay and check recording),
source tracing, and an about page explaining how the system works and its limits.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from contamination.diagnosis import load_knowledge
from contamination.pipeline import diagnose_batch_as_of, fit, load_campaign, run_campaign
from contamination.schema import INVESTIGATION_COLUMNS, VARIABLES

OBSERVED = "#2a78d6"
BAND = "rgba(140, 140, 135, 0.22)"
MEDIAN = "#8a8984"
ALERT = "#ec835a"
STRONG = "#eb6834"
POSSIBLE = "#9e9d98"
CASE = "#2a78d6"

st.set_page_config(page_title="Contamination investigation", page_icon="🧫", layout="wide")


# --------------------------------------------------------------------------- data


@st.cache_resource(show_spinner="Loading campaign and fitting the reference profile…")
def get_campaign(data_dir: str, n_reference: int):
    campaign = load_campaign(data_dir, n_reference)
    return campaign, fit(campaign)


@st.cache_resource(show_spinner="Detecting, diagnosing and tracing sources…")
def get_result(data_dir: str, n_reference: int, investigations_csv: str):
    campaign, reference = get_campaign(data_dir, n_reference)
    return run_campaign(campaign, reference, _inv_from_csv(investigations_csv))


def _inv_from_csv(text: str) -> pd.DataFrame:
    df = (
        pd.read_csv(io.StringIO(text), dtype={"note": "object"}) if text.strip() else pd.DataFrame()
    )
    return df if len(df) else pd.DataFrame(columns=INVESTIGATION_COLUMNS)


def investigations() -> pd.DataFrame:
    return _inv_from_csv(st.session_state["investigations_csv"])


# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("🧫 Contamination investigation")
    data_dir = st.text_input("Campaign folder", "data/synthetic")
    n_reference = st.number_input("Approved reference batches (earliest)", 5, 200, 30)
    if not (Path(data_dir) / "measurements.csv").exists():
        st.error(f"No measurements.csv in `{data_dir}`. Run `python -m contamination.demo` first.")
        st.stop()
    campaign, reference = get_campaign(data_dir, int(n_reference))
    if (
        "investigations_csv" not in st.session_state
        or st.session_state.get("loaded_from") != data_dir
    ):
        st.session_state["investigations_csv"] = campaign.investigations.to_csv(index=False)
        st.session_state["loaded_from"] = data_dir
    page = st.radio(
        "View", ["Campaign overview", "Batch investigation", "Source tracing", "How it works"]
    )
    show_truth = False
    if campaign.outcomes is not None:
        show_truth = st.toggle(
            "Show simulated ground truth",
            value=False,
            help="Evaluation only. The model never sees it.",
        )
    st.caption(f"Reference `{reference.version}` · {len(reference.batch_ids)} batches")

kb = load_knowledge()
result = get_result(data_dir, int(n_reference), st.session_state["investigations_csv"])
synthetic = (
    campaign.outcomes is not None and (campaign.outcomes.get("data_origin") == "synthetic").any()
)
if synthetic:
    st.warning(
        "**Synthetic data**: a software demonstration on simulated batches, not real contamination "
        "detection. The knowledge base is a draft that MaverickX scientists have not reviewed.",
        icon="⚠️",
    )


def truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    if show_truth and campaign.outcomes is not None:
        return df.merge(campaign.outcomes[["batch_id", "scenario"]], on="batch_id", how="left")
    return df


# --------------------------------------------------------------------------- charts


def batch_figure(batch_id: str, scores: pd.DataFrame, alerts, as_of_h: float | None) -> go.Figure:
    variables = list(VARIABLES)
    fig = make_subplots(
        rows=len(variables),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f"{v} ({VARIABLES[v].unit})" for v in variables],
    )
    s = scores[scores["batch_id"] == batch_id]
    for i, var in enumerate(variables, 1):
        ref = reference.stats[reference.stats["variable"] == var].sort_values("hour")
        lo, hi = ref["median"] - 3 * ref["scale"], ref["median"] + 3 * ref["scale"]
        first = i == 1
        fig.add_trace(
            go.Scatter(
                x=ref["hour"],
                y=lo,
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            ),
            i,
            1,
        )
        fig.add_trace(
            go.Scatter(
                x=ref["hour"],
                y=hi,
                mode="lines",
                fill="tonexty",
                fillcolor=BAND,
                line={"width": 0},
                name="Reference ±3 robust SD",
                legendgroup="band",
                showlegend=first,
                hoverinfo="skip",
            ),
            i,
            1,
        )
        fig.add_trace(
            go.Scatter(
                x=ref["hour"],
                y=ref["median"],
                mode="lines",
                line={"color": MEDIAN, "width": 1, "dash": "dash"},
                name="Reference median",
                legendgroup="median",
                showlegend=first,
                hovertemplate="ref median %{y:.3g} at %{x} h<extra></extra>",
            ),
            i,
            1,
        )
        g = s[s["variable"] == var].sort_values("elapsed_h")
        x, y = g["elapsed_h"].to_numpy(float), g["value"].to_numpy(float)
        if len(x) > 2:  # break the line across gaps of missing readings
            step = np.median(np.diff(x))
            gaps = np.where(np.diff(x) > 1.5 * step)[0] + 1
            x, y = np.insert(x, gaps, np.nan), np.insert(y, gaps, np.nan)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers" if len(g) < 20 else "lines",
                line={"color": OBSERVED, "width": 2},
                marker={"size": 7},
                name="Observed",
                legendgroup="obs",
                showlegend=first,
                hovertemplate="%{y:.3g} at %{x} h<extra>observed</extra>",
            ),
            i,
            1,
        )
        flagged = g[
            [
                any(a.variable == var and a.start_h <= h <= a.end_h for a in alerts)
                for h in g["elapsed_h"]
            ]
        ]
        if len(flagged):
            fig.add_trace(
                go.Scatter(
                    x=flagged["elapsed_h"],
                    y=flagged["value"],
                    mode="markers",
                    marker={"color": ALERT, "size": 9, "line": {"color": "white", "width": 2}},
                    name="Alert",
                    legendgroup="alert",
                    showlegend=not any(t.name == "Alert" for t in fig.data),
                    customdata=flagged["z"],
                    hovertemplate="%{y:.3g} at %{x} h · robust z %{customdata:+.1f}<extra>alert</extra>",
                ),
                i,
                1,
            )
        if as_of_h is not None:
            fig.add_vline(
                x=as_of_h, line={"color": MEDIAN, "width": 1, "dash": "dot"}, row=i, col=1
            )
    fig.update_layout(
        height=230 * len(variables),
        margin={"l": 10, "r": 10, "t": 80, "b": 10},
        legend={"orientation": "h", "y": 0.995, "yanchor": "top", "x": 0, "yref": "container"},
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Elapsed time (h)", row=len(variables), col=1)
    return fig


def genealogy_figure(trace) -> go.Figure:
    leads = [c for c in trace.candidates if c.tier in ("strong lead", "possible lead")]
    cases = sorted({b for c in leads for b in c.cases})
    status = trace.statuses.set_index("batch_id")["status"]
    # Group resources by cluster (largest first) so linked cases sit together.
    order = [
        r for cl in sorted(trace.clusters, key=lambda c: -len(c["cases"])) for r in cl["linked_by"]
    ]
    leads.sort(
        key=lambda c: (order.index(c.resource) if c.resource in order else len(order), c.resource)
    )
    ypos_r = {c.resource: i for i, c in enumerate(leads)}
    # Order cases by the average position of their resources to reduce line crossings.
    cases.sort(key=lambda b: (np.mean([ypos_r[c.resource] for c in leads if b in c.cases]), b))
    ypos_b = {b: i * (max(len(leads) - 1, 1) / max(len(cases) - 1, 1)) for i, b in enumerate(cases)}
    fig = go.Figure()
    for c in leads:
        for b in c.cases:
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[ypos_r[c.resource], ypos_b[b]],
                    mode="lines",
                    line={"color": STRONG if c.tier == "strong lead" else POSSIBLE, "width": 2},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    for tier, color in (("strong lead", STRONG), ("possible lead", POSSIBLE)):
        sel = [c for c in leads if c.tier == tier]
        if sel:
            fig.add_trace(
                go.Scatter(
                    x=[0] * len(sel),
                    y=[ypos_r[c.resource] for c in sel],
                    mode="markers+text",
                    marker={"size": 16, "color": color, "line": {"color": "white", "width": 2}},
                    text=[c.resource for c in sel],
                    textposition="middle left",
                    name=tier.capitalize(),
                    customdata=[c.explanation for c in sel],
                    hovertemplate="%{text}<br>%{customdata}<extra></extra>",
                )
            )
    fig.add_trace(
        go.Scatter(
            x=[1] * len(cases),
            y=[ypos_b[b] for b in cases],
            mode="markers+text",
            marker={"size": 14, "color": CASE, "line": {"color": "white", "width": 2}},
            text=cases,
            textposition="middle right",
            name="Contamination case",
            customdata=[status.get(b, "") for b in cases],
            hovertemplate="%{text}: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(320, 60 * max(len(leads), len(cases))),
        margin={"l": 150, "r": 120, "t": 20, "b": 20},
        xaxis={"visible": False, "range": [-0.6, 1.5]},
        yaxis={"visible": False},
        legend={"orientation": "h", "y": 1.08},
    )
    return fig


# --------------------------------------------------------------------------- pages


def page_overview():
    st.header("Campaign overview")
    ov = result.overview(campaign)
    flagged = ov[ov["process_deviation"] != "within reference"]
    c = st.columns(5)
    c[0].metric("Batches monitored", len(ov))
    c[1].metric("Flagged for review", len(flagged))
    c[2].metric(
        "Contamination suspected", int(ov["status"].str.startswith("contamination suspected").sum())
    )
    c[3].metric(
        "Confirmed by test", int(ov["status"].str.startswith("contamination confirmed").sum())
    )
    c[4].metric(
        "Strong source leads", sum(t.tier == "strong lead" for t in result.trace.candidates)
    )

    only_flagged = st.toggle("Only flagged batches", value=True)
    table = truth_columns(flagged if only_flagged else ov)
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "start": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
            "first_alert_h": st.column_config.NumberColumn("first alert (h)", format="%.1f"),
        },
    )
    st.caption(
        "Process deviation and contamination evidence are separate: a batch can be strongly abnormal "
        "with no contamination confirmed. Rankings are for review, not probabilities."
    )


def page_batch():
    st.header("Batch investigation")
    ov = result.overview(campaign)
    flagged = ov.loc[ov["process_deviation"] != "within reference", "batch_id"].tolist()
    options = flagged + [b for b in campaign.monitored_ids if b not in flagged]
    left, right = st.columns([1, 2])
    batch_id = left.selectbox(
        "Batch", options, format_func=lambda b: f"{b}{' ⚑' if b in flagged else ''}"
    )
    m = campaign.measurements[campaign.measurements["batch_id"] == batch_id]
    max_h = float(np.ceil(m["available_h"].max()))
    replay = right.toggle("Replay: view as of a point in the batch", value=False)
    as_of = right.slider("As of elapsed hour", 0.0, max_h, max_h, 0.5) if replay else None

    inv = investigations()
    diag = diagnose_batch_as_of(campaign, reference, batch_id, as_of, inv)
    report = diag.reports.get(batch_id)
    scores, alerts = diag.monitor_result.scores, diag.monitor_result.alerts
    full_report = result.diagnosis.reports.get(batch_id)
    if report is not None and as_of is None and full_report is not None:
        report.sources, report.assessment["affected_scope"] = (
            full_report.sources,
            full_report.assessment["affected_scope"],
        )

    if show_truth and campaign.outcomes is not None:
        o = campaign.outcomes.set_index("batch_id").loc[batch_id]
        st.info(f"Simulated ground truth: **{o['scenario']}**: {o['reason']}")
    st.plotly_chart(
        batch_figure(batch_id, scores, alerts, as_of), use_container_width=True, theme="streamlit"
    )

    if report is None:
        st.success(
            "No alerts"
            + (
                f" as of {as_of:g} h."
                if as_of is not None
                else ": the batch stayed within the reference profile."
            )
        )
        return

    a = report.assessment
    lead_name = report.leading.name if report.leading else "none"
    cards = [
        ("Leading explanation", lead_name),
        ("Contamination evidence", a["contamination_evidence"]),
        ("Process deviation", a["process_deviation"]),
        ("Confidence", a["confidence"]),
    ]
    for col, (label, value) in zip(st.columns(4), cards):
        with col.container(border=True):
            st.caption(label)
            st.markdown(f"**{value}**")
    st.markdown(
        f"**Quality impact:** {a['quality_impact']}  \n**Affected scope:** {a['affected_scope']}"
    )

    t1, t2, t3, t4, t5 = st.tabs(
        ["Explanations", "Alerts", "Evidence", "Next checks & sources", "Full report"]
    )
    with t1:
        hyp = pd.DataFrame(
            [
                {
                    "id": h.id,
                    "explanation": h.name,
                    "tier": h.tier,
                    "score": h.score,
                    "supporting": ", ".join(f"{e} {report.evidence[e].name}" for e in h.supporting),
                    "contradicting": ", ".join(
                        f"{e} {report.evidence[e].name}" for e in h.contradicting
                    ),
                    "missing": ", ".join(h.missing),
                }
                for h in report.hypotheses
            ]
        )
        st.dataframe(hyp, hide_index=True, use_container_width=True)
        st.caption(
            "Score = supporting weights − contradicting weights. It orders explanations for review; it is not a probability."
        )
    with t2:
        al = pd.DataFrame([x.to_dict() for x in alerts])
        cols = [
            "variable",
            "detector",
            "direction",
            "start_h",
            "end_h",
            "first_raised_h",
            "persistence",
            "observed_at_peak",
            "reference_median_at_peak",
            "robust_z_at_peak",
            "corroborating_variables",
            "data_quality_warnings",
        ]
        st.dataframe(al[cols], hide_index=True, use_container_width=True)
    with t3:
        ev = pd.DataFrame(
            [
                {"id": e.id, "evidence": e.name, "state": e.state, "detail": e.detail}
                for e in report.evidence.values()
            ]
        )
        st.dataframe(ev, hide_index=True, use_container_width=True)
    with t4:
        st.subheader("Recommended next checks")
        for i, ch in enumerate(report.next_checks, 1):
            st.markdown(
                f"{i}. **{ch['name']}** (~{ch['turnaround_h']:g} h, {ch['relative_cost']} cost): {ch['reason']}"
            )
        st.subheader("Possible sources (leads, not proof)")
        if as_of is not None:
            st.caption("Source tracing uses the end-of-batch statuses of the whole campaign.")
        if not report.sources:
            st.write("None: source tracing not run, or contamination not suspected.")
        for src in report.sources:
            st.markdown(
                f"- **{src['resource']}** ({src['tier']}): {src['explanation']}  \n  *Suggested check:* {src['suggested_check']}"
            )
        st.subheader("Record a check result")
        with st.form("record_check", clear_on_submit=True):
            f1, f2, f3 = st.columns(3)
            check = f1.selectbox(
                "Check", list(kb.checks), format_func=lambda c: f"{c}: {kb.checks[c]['name']}"
            )
            res = f2.selectbox("Result", ["positive", "negative", "inconclusive"])
            hour = f3.number_input(
                "Result available at (elapsed h)",
                0.0,
                200.0,
                float(as_of if as_of is not None else max_h),
            )
            note = st.text_input("Note")
            if st.form_submit_button("Add result"):
                row = pd.DataFrame(
                    [[batch_id, check, res, hour, note or None]], columns=INVESTIGATION_COLUMNS
                )
                st.session_state["investigations_csv"] = pd.concat(
                    [inv, row], ignore_index=True
                ).to_csv(index=False)
                st.rerun()
        batch_inv = inv[inv["batch_id"] == batch_id]
        if len(batch_inv):
            st.dataframe(batch_inv, hide_index=True, use_container_width=True)
        st.download_button(
            "Download all recorded checks (CSV)",
            st.session_state["investigations_csv"],
            "investigations.csv",
            "text/csv",
        )
    with t5:
        md = report.to_markdown()
        st.download_button(
            "Download report (Markdown)", md, f"{batch_id}_report.md", "text/markdown"
        )
        st.markdown(md)


def page_tracing():
    st.header("Source tracing")
    st.write(
        "Each shared input or piece of equipment is compared between batches that used it and batches that "
        "did not. A resource is a **strong lead** when contamination cases are clearly over-represented among "
        "its users (one-sided Fisher's exact test, Bonferroni-corrected, confounding-checked). "
        "**A shared resource is a lead for investigation, not proof of cause.**"
    )
    trace = result.trace
    if not trace.candidates:
        st.info("No contamination cases, so there is nothing to trace.")
        return
    st.plotly_chart(genealogy_figure(trace), use_container_width=True, theme="streamlit")
    for i, c in enumerate(trace.clusters, 1):
        st.markdown(
            f"**Cluster {i}:** cases {', '.join(c['cases'])}, linked by {', '.join(c['linked_by'])}"
        )
    cols = [
        "resource",
        "tier",
        "cases",
        "non_cases",
        "attack_rate",
        "attack_rate_elsewhere",
        "relative_risk",
        "p_value",
        "p_adjusted",
        "suggested_check",
        "explanation",
    ]
    st.dataframe(
        trace.table()[cols],
        hide_index=True,
        use_container_width=True,
        column_config={
            "attack_rate": st.column_config.NumberColumn(format="percent"),
            "attack_rate_elsewhere": st.column_config.NumberColumn(format="percent"),
            "p_value": st.column_config.NumberColumn(format="%.2g"),
            "p_adjusted": st.column_config.NumberColumn(format="%.2g"),
        },
    )
    with st.expander("Batch statuses used for tracing"):
        st.dataframe(truth_columns(trace.statuses), hide_index=True, use_container_width=True)


def page_about():
    st.header("How it works")
    st.markdown(
        """
The system helps scientists investigate contamination faster. **It supports decisions; it does not make them.**

1. **Check the data.** Readings that are missing, implausible, in the wrong unit or duplicated are flagged, never silently deleted or filled with zero.
2. **Learn what normal looks like.** From batches scientists approve as normal, it learns the typical value of each measurement at each hour, and how much it usually varies. This *reference profile* is then frozen and versioned.
3. **Spot unusual behaviour.** Each new reading is compared with the reference. Three simple detectors raise alerts: a single reading far from normal, a sustained drift (EWMA) and an accumulating shift (CUSUM). Every alert shows what changed, by how much, and when.
4. **Suggest explanations.** A reviewable rule book (`knowledge/catalogue.yaml`) links patterns of evidence to explanations: contamination, slow growth, a raw-material/condition change, or a sensor fault. Explanations are ranked by a transparent score. When the evidence is thin, the answer is *"unresolved, too early to tell"*.
5. **Recommend the next check.** It suggests the quickest, cheapest check that best separates the leading explanations, and keeps a contamination check on the list until contamination is ruled out.
6. **Trace possible sources.** It asks what contaminated batches had in common (medium lot, inoculum lot, bioreactor) compared with clean batches.
7. **Learn from confirmed results.** Recording a test result (e.g. a positive microscopy result) updates the assessment. Only an independent test can mark contamination as *confirmed*.
        """
    )
    st.subheader("Limits")
    st.markdown(
        f"""
- Built and evaluated on **synthetic** α-amylase batches; real performance is unknown until tested on MaverickX data.
- The rule book is a **draft** ({kb.review_status}).
- Anomaly scores are not contamination probabilities, and no organism is identified from process data.
- Source tracing only knows the recorded genealogy (medium lot, inoculum lot, bioreactor).
        """
    )


{
    "Campaign overview": page_overview,
    "Batch investigation": page_batch,
    "Source tracing": page_tracing,
    "How it works": page_about,
}[page]()
