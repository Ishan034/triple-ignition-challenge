"""Contamination investigation dashboard (Streamlit).

    streamlit run app/dashboard.py

Pages: campaign overview, batch investigation (time replay, how the assessment
developed, check results and alert feedback), detection timing, source tracing, a value
estimate from the plant's own figures, loading a plant's own data, and an about page.
"""

from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from pathlib import Path

# Import the model from this checkout's src/, not from an installed copy. Streamlit
# Cloud installs the package only when requirements.txt changes, and it keeps the
# server process (with the modules it already imported) running across pushes, so the
# model in memory can lag behind the code that was pushed (an ImportError on new names).
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SRC_STAMP = max(p.stat().st_mtime for p in (SRC / "contamination").rglob("*.py"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

_loaded = sys.modules.get("contamination")
if _loaded is not None and (
    not str(getattr(_loaded, "__file__", "")).startswith(str(SRC))
    or getattr(_loaded, "_src_stamp", None) != SRC_STAMP
):
    # A stale model is loaded in this process: drop it, and the results cached with it.
    for name in [m for m in sys.modules if m == "contamination" or m.startswith("contamination.")]:
        del sys.modules[name]
    st.cache_resource.clear()
    st.cache_data.clear()

import contamination
from contamination.evaluate import per_batch
from contamination.importer import convert, readiness
from contamination.pipeline import (
    diagnose_batch_as_of,
    diagnosis_timeline,
    first_named,
    fit_cached,
    load_campaign,
    measurements_path,
    run_campaign,
)
from contamination.schema import INVESTIGATION_COLUMNS, VARIABLES
from contamination.value import ValueInputs, estimate

contamination._src_stamp = SRC_STAMP

OBSERVED = "#2a78d6"
BAND = "rgba(140, 140, 135, 0.22)"
MEDIAN = "#8a8984"
ALERT = "#ec835a"
STRONG = "#eb6834"
POSSIBLE = "#9e9d98"
CASE = "#2a78d6"
TEMPLATE_DIR = Path("docs/data-template")
IMPORT_ROOT = Path("data/imported")
FEEDBACK_COLUMNS = ["batch_id", "variable", "detector", "start_h", "verdict", "note"]
DATASETS = {
    "Synthetic demo (simulated contamination)": "data/synthetic",
    "IndPenSim (simulated penicillin, equipment faults)": "data/indpensim",
    "Erythromycin plant (real batches, no labels)": "data/erythromycin",
}
OTHER = "Other folder…"
BANNERS = {
    "synthetic": (
        "**Synthetic data**: a software demonstration on simulated batches, not real contamination "
        "detection. The knowledge base is a draft that MaverickX scientists have not reviewed."
    ),
    "IndPenSim": (
        "**IndPenSim**: a published high-fidelity simulation of an industrial 100,000 L penicillin "
        "process (Goldrick et al. 2015, CC BY 4.0). Batches P091–P100 contain labelled **equipment "
        "faults, not contamination**: any contamination call here would be wrong. Known outcomes "
        "show the fault type and when it started."
    ),
    "Erythromycin": (
        "**Erythromycin plant**: real hourly data from 406 industrial batches (Zenodo 14619074, "
        "CC BY 4.0). There are **no fault or contamination labels**; final potency is held out as "
        "the outcome. Flags mean *worth review*, not confirmed problems."
    ),
}

st.set_page_config(page_title="Contamination investigation", page_icon="🧫", layout="wide")


# --------------------------------------------------------------------------- data


@st.cache_resource(
    show_spinner="Loading the campaign and fitting the reference profile "
    "(the first time on a large dataset this takes a few minutes)…"
)
def get_campaign(data_dir: str, n_reference: int):
    campaign = load_campaign(data_dir, n_reference)
    return campaign, fit_cached(campaign)


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


def feedback() -> pd.DataFrame:
    text = st.session_state.get("feedback_csv", "")
    df = (
        pd.read_csv(io.StringIO(text), dtype={"note": "object"}) if text.strip() else pd.DataFrame()
    )
    return df if len(df) else pd.DataFrame(columns=FEEDBACK_COLUMNS)


@st.cache_data(show_spinner="Replaying the batch hour by hour…")
def get_timeline(data_dir: str, n_reference: int, batch_id: str, investigations_csv: str):
    campaign, reference = get_campaign(data_dir, n_reference)
    return diagnosis_timeline(campaign, reference, batch_id, _inv_from_csv(investigations_csv))


# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("🧫 Contamination investigation")
    choices = {k: v for k, v in DATASETS.items() if measurements_path(v)}
    for folder in sorted(IMPORT_ROOT.glob("*")) if IMPORT_ROOT.exists() else []:
        if measurements_path(folder):
            choices[f"Imported: {folder.name}"] = str(folder)
    wanted = st.session_state.get("data_dir")
    labels = list(choices) + [OTHER]
    index = next((i for i, k in enumerate(labels) if choices.get(k) == wanted), 0)
    dataset = st.selectbox("Dataset", labels, index=index)
    data_dir = (
        st.text_input("Campaign folder", wanted or "data/synthetic")
        if dataset == OTHER
        else choices[dataset]
    )
    if measurements_path(data_dir) is None:
        st.error(f"No measurements.csv in `{data_dir}`. Run `python -m contamination.demo` first.")
        st.stop()
    has_list = (Path(data_dir) / "reference_batches.txt").exists()
    n_reference = (
        30 if has_list else st.number_input("Approved reference batches (earliest)", 5, 1000, 30)
    )
    campaign, reference = get_campaign(data_dir, int(n_reference))
    if (
        "investigations_csv" not in st.session_state
        or st.session_state.get("loaded_from") != data_dir
    ):
        st.session_state["investigations_csv"] = campaign.investigations.to_csv(index=False)
        st.session_state["feedback_csv"] = ""
        st.session_state["loaded_from"] = data_dir
    page = st.radio(
        "View",
        [
            "Campaign overview",
            "Batch investigation",
            "Detection timing",
            "Source tracing",
            "Value estimate",
            "Load your data",
            "How it works",
        ],
    )
    show_truth = False
    if campaign.outcomes is not None:
        show_truth = st.toggle(
            "Show known outcomes",
            value=False,
            help="Evaluation only (simulated truth, fault labels or final potency). "
            "The model never sees it.",
        )
    st.caption(
        f"Reference `{reference.version}` · {len(reference.batch_ids)} approved batches "
        f"({campaign.reference_source}) · {len(campaign.monitored_ids)} monitored"
    )

kb = campaign.kb
result = get_result(data_dir, int(n_reference), st.session_state["investigations_csv"])
origin = (
    " ".join(campaign.outcomes["data_origin"].dropna().astype(str).unique())
    if campaign.outcomes is not None and "data_origin" in campaign.outcomes
    else ""
)
synthetic = origin == "synthetic"
for key, text in BANNERS.items():
    if key in origin:
        (st.warning if key == "synthetic" else st.info)(
            text, icon="⚠️" if key == "synthetic" else "ℹ️"
        )


def truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    if show_truth and campaign.outcomes is not None:
        return df.merge(campaign.outcomes[["batch_id", "scenario"]], on="batch_id", how="left")
    return df


# --------------------------------------------------------------------------- charts


def batch_figure(batch_id: str, scores: pd.DataFrame, alerts, as_of_h: float | None) -> go.Figure:
    present = set(campaign.measurements["variable"])
    variables = [v for v in VARIABLES if v in present]
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
    fb = feedback()
    if len(fb):
        counts = fb["verdict"].value_counts()
        st.markdown(
            f"**Scientist feedback on alerts:** {len(fb)} reviewed · "
            + " · ".join(f"{k}: {v}" for k, v in counts.items())
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
        extra = (
            f" · final potency {o['final_potency']:,.0f}"
            if "final_potency" in o and pd.notna(o["final_potency"])
            else ""
        )
        st.info(f"Known outcome: **{o['scenario']}**: {o['reason']}{extra}")
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

    t1, t2, t3, t4, t6, t7, t5 = st.tabs(
        [
            "Explanations",
            "Alerts",
            "Evidence",
            "Next checks & sources",
            "How it developed",
            "Alert feedback",
            "Full report",
        ]
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
    with t6:
        timeline_section(batch_id)
    with t7:
        feedback_section(batch_id, alerts)
    with t5:
        md = report.to_markdown()
        st.download_button(
            "Download report (Markdown)", md, f"{batch_id}_report.md", "text/markdown"
        )
        st.markdown(md)


def timeline_section(batch_id: str):
    st.caption(
        "The batch replayed hour by hour, using only the results available at each hour: "
        "when the first alert came, and when each explanation first became the leading one."
    )
    tl = get_timeline(data_dir, int(n_reference), batch_id, st.session_state["investigations_csv"])
    if tl.empty:
        st.write("No alerts in this batch.")
        return
    fig = go.Figure(
        go.Scatter(
            x=tl["hour"],
            y=tl["leading_explanation"],
            mode="lines+markers",
            line={"shape": "hv", "color": OBSERVED},
            hovertemplate="%{x:g} h: %{y}<extra></extra>",
            name="Leading explanation",
        )
    )
    marks = [("first alert", float(tl["hour"].min()), ALERT)]
    if show_truth and campaign.outcomes is not None:
        o = campaign.outcomes.set_index("batch_id").loc[batch_id]
        if pd.notna(o["onset_h"]):
            marks.append(("true onset (simulated)", float(o["onset_h"]), MEDIAN))
        if pd.notna(o["reference_event_h"]):
            marks.append(("today's method would notice", float(o["reference_event_h"]), STRONG))
    for label, x, color in marks:
        fig.add_vline(
            x=x,
            line={"color": color, "dash": "dot"},
            annotation_text=label,
            annotation_position="top",
        )
    fig.update_layout(
        height=320, margin={"l": 10, "r": 10, "t": 40, "b": 10}, xaxis_title="Elapsed time (h)"
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    named = first_named(tl)
    if len(named):
        st.dataframe(
            named.rename(columns={"hour": "first leading at (h)"}),
            hide_index=True,
            use_container_width=True,
        )


def feedback_section(batch_id: str, alerts):
    st.caption(
        "Mark whether an alert pointed at a real issue. Feedback is kept apart from the model: it "
        "measures false alarms on your own batches and is the basis for re-tuning the limits."
    )
    if not alerts:
        st.write("No alerts to review.")
        return
    labels = {
        f"{a.variable} · {a.detector} · {a.start_h:g}-{a.end_h:g} h ({a.direction})": a
        for a in sorted(alerts, key=lambda a: a.start_h)
    }
    with st.form("alert_feedback", clear_on_submit=True):
        choice = st.selectbox("Alert", list(labels))
        verdict = st.radio("Verdict", ["real issue", "false alarm", "not sure"], horizontal=True)
        note = st.text_input("Note (what was found)")
        if st.form_submit_button("Save feedback"):
            a = labels[choice]
            row = pd.DataFrame(
                [[batch_id, a.variable, a.detector, a.start_h, verdict, note or None]],
                columns=FEEDBACK_COLUMNS,
            )
            st.session_state["feedback_csv"] = pd.concat(
                [feedback(), row], ignore_index=True
            ).to_csv(index=False)
            st.rerun()
    fb = feedback()
    if len(fb[fb["batch_id"] == batch_id]):
        st.dataframe(fb[fb["batch_id"] == batch_id], hide_index=True, use_container_width=True)
    if len(fb):
        st.download_button(
            "Download all alert feedback (CSV)",
            fb.to_csv(index=False),
            "alert_feedback.csv",
            "text/csv",
        )


def campaign_timing() -> pd.DataFrame | None:
    """Per batch with a known onset: hours to the first alert and lead over today's method."""
    if campaign.outcomes is None:
        return None
    o = campaign.outcomes[campaign.outcomes["batch_id"].isin(campaign.monitored_ids)]
    pb = per_batch(result.diagnosis.monitor_result.alerts, o)
    return pb[pb["category"] != "normal"]


def page_timing():
    st.header("Detection timing")
    st.write(
        "How early the system warns. **Onset → first alert** measures speed; **lead over today's "
        "method** measures the time it saves. Both only mean something together with the false-alarm "
        "rate, because tighter limits always alert sooner."
    )
    pb = campaign_timing()
    if pb is None or pb["onset_to_alert_h"].isna().all():
        st.info(
            "Timing needs known outcomes with an onset time (when the problem started or was first "
            "noticed). Ask for these with the batch records: see the data request."
        )
        return
    o = campaign.outcomes[campaign.outcomes["batch_id"].isin(campaign.monitored_ids)]
    normal = per_batch(result.diagnosis.monitor_result.alerts, o[o["category"] == "normal"])
    cont = pb[pb["category"] == "contamination"]
    c = st.columns(4)
    c[0].metric("Problem batches caught", f"{int(pb['flagged'].sum())}/{len(pb)}")
    c[1].metric("Median onset → first alert", f"{pb['onset_to_alert_h'].median():.1f} h")
    lead = cont["lead_time_vs_reference_h"].median() if len(cont) else np.nan
    c[2].metric("Median lead over today's method", "n/a" if np.isnan(lead) else f"{lead:.1f} h")
    c[3].metric(
        "Normal batches with any alert",
        f"{int(normal['first_alert_any_h'].notna().sum())}/{len(normal)}",
    )
    order = pb.sort_values("onset_to_alert_h")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=order["batch_id"],
            x=order["onset_to_alert_h"],
            orientation="h",
            marker_color=OBSERVED,
            name="onset → first alert (h)",
            customdata=order["scenario"],
            hovertemplate="%{y} (%{customdata}): %{x:.1f} h<extra></extra>",
        )
    )
    if order["lead_time_vs_reference_h"].notna().any():
        fig.add_trace(
            go.Bar(
                y=order["batch_id"],
                x=order["lead_time_vs_reference_h"],
                orientation="h",
                marker_color=STRONG,
                name="lead over today's method (h)",
                hovertemplate="%{y}: %{x:.1f} h earlier<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        height=max(300, 26 * len(order)),
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title="hours",
        legend={"orientation": "h", "y": 1.05},
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.dataframe(pb, hide_index=True, use_container_width=True)
    st.caption("Uses the outcomes table (known answers): evaluation only, never seen by the model.")
    if synthetic:
        st.caption(
            "Synthetic data: the onset and today's method are simulated, so these hours show the software "
            "works as designed, not real-world timing."
        )


def page_value():
    st.header("Value estimate")
    st.write(
        "A yearly estimate from **your plant's own figures**. The system's contribution is limited to "
        "what evaluation measured (how much earlier it warns, how many false alarms it raises). Every "
        "cost and rate below must come from the plant."
    )
    st.warning(
        "The pre-filled values are placeholders for the demo, not MaverickX figures. "
        "Replace them before using the result.",
        icon="⚠️",
    )
    pb = campaign_timing()
    default_lead = 0.0
    if pb is not None:
        cont = pb[pb["category"] == "contamination"]
        if cont["lead_time_vs_reference_h"].notna().any():
            default_lead = float(round(cont["lead_time_vs_reference_h"].median(), 1))
    summary = result.diagnosis.monitor_result.summary
    flagged = set(summary.loc[summary["n_alerts"] > 0, "batch_id"])
    if campaign.outcomes is not None:
        normal = campaign.outcomes.loc[
            (campaign.outcomes["category"] == "normal")
            & campaign.outcomes["batch_id"].isin(campaign.monitored_ids),
            "batch_id",
        ]
        default_fa = float(round(normal.isin(flagged).mean(), 2)) if len(normal) else 0.1
    else:
        default_fa = float(round(len(flagged) / max(len(campaign.monitored_ids), 1), 2))
    a, b = st.columns(2)
    with a:
        st.subheader("Your plant")
        batches = st.number_input("Batches per year", 1.0, 10000.0, 100.0, 10.0)
        rate = st.number_input("Share lost to contamination today (%)", 0.0, 100.0, 5.0, 0.5) / 100
        lost = st.number_input("Cost of one lost batch", 0.0, 1e8, 50000.0, 1000.0)
        hourly = st.number_input("Cost per hour of running a doomed batch", 0.0, 1e6, 500.0, 50.0)
        test_cost = st.number_input(
            "Cost of one expensive confirmatory test", 0.0, 1e6, 10000.0, 500.0
        )
        fa_cost = st.number_input("Cost of checking one false alarm", 0.0, 1e6, 200.0, 50.0)
    with b:
        st.subheader("What the system changes")
        hours = st.number_input(
            "Hours earlier than today's method",
            0.0,
            200.0,
            default_lead,
            0.5,
            help="Pre-filled from this campaign's evaluation where outcomes exist.",
        )
        shared = st.slider("Share of cases from a shared source (%)", 0, 100, 50) / 100
        spared = st.number_input(
            "Later batches spared once a shared source is found", 0.0, 50.0, 1.0, 0.5
        )
        tests = st.number_input("Expensive tests avoided per case", 0.0, 20.0, 1.0, 0.5)
        good = st.number_input(
            "Good batches not scrapped per year (sensor fault recognised)", 0.0, 100.0, 0.0, 1.0
        )
        fa = (
            st.slider(
                "Normal batches that raise an alert needing a check (%)",
                0,
                100,
                round(100 * default_fa),
                help="Pre-filled from this campaign: normal batches with any alert (all flagged "
                "batches when outcomes are unknown).",
            )
            / 100
        )
    table = estimate(
        ValueInputs(
            batches, rate, lost, hourly, hours, shared, spared, tests, test_cost, good, fa, fa_cost
        )
    )
    net = table["value_per_year"].iloc[-1]
    st.metric("Net value per year (input currency)", f"{net:,.0f}")
    parts = table.iloc[:-1]
    fig = go.Figure(
        go.Bar(
            x=parts["value_per_year"],
            y=parts["mechanism"],
            orientation="h",
            marker_color=[STRONG if v < 0 else OBSERVED for v in parts["value_per_year"]],
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=300, margin={"l": 10, "r": 10, "t": 10, "b": 10}, yaxis={"autorange": "reversed"}
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={"value_per_year": st.column_config.NumberColumn(format="%,.0f")},
    )


def _template_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(TEMPLATE_DIR.glob("*")):
            if f.is_file():
                z.write(f, f.name)
    return buf.getvalue()


def page_load():
    st.header("Load your data")
    st.write(
        "Upload your plant's exports (CSV or Excel) together with a `mapping.yaml` that says which file and "
        "column holds what. The data is converted, checked, and can then be monitored here. Values are copied "
        "as they are: nothing is filled, rescaled or deleted."
    )
    if TEMPLATE_DIR.exists():
        st.download_button(
            "Download the data template (example exports + mapping.yaml)",
            _template_zip(),
            "data-template.zip",
            "application/zip",
        )
    name = st.text_input("Name for this dataset", "maverickx")
    files = st.file_uploader(
        "Exports and mapping.yaml",
        accept_multiple_files=True,
        type=["csv", "xlsx", "xls", "yaml", "yml"],
    )
    if files and st.button("Convert and check"):
        if not any(f.name in ("mapping.yaml", "mapping.yml") for f in files):
            st.error("Include a mapping.yaml (see the template).")
            return
        raw = Path(tempfile.mkdtemp(prefix="upload_"))
        for f in files:
            (raw / Path(f.name).name).write_bytes(f.getvalue())
        out = IMPORT_ROOT / "".join(ch for ch in name if ch.isalnum() or ch in "-_")
        mapping = raw / ("mapping.yaml" if (raw / "mapping.yaml").exists() else "mapping.yml")
        try:
            res = convert(mapping, out)
        except (ValueError, KeyError) as e:
            st.error(f"Could not convert: {e}")
            return
        st.success(
            f"Converted into `{out}`: "
            + ", ".join(f"{k} {len(v)} rows" for k, v in res.tables.items())
        )
        if res.unmapped:
            with st.expander(f"{len(res.unmapped)} columns or names not used by the mapping"):
                st.write(res.unmapped)
        st.session_state["imported_dir"] = str(out)
    target = st.session_state.get("imported_dir")
    if target:
        st.subheader("Readiness")
        r = readiness(target)
        icon = {"ok": "✅", "warn": "⚠️", "missing": "❌"}
        r.insert(0, "", r["status"].map(icon))
        st.dataframe(r, hide_index=True, use_container_width=True)
        if st.button(f"Monitor `{target}` now"):
            st.session_state["data_dir"] = target
            st.rerun()


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
3. **Spot unusual behaviour.** Each new reading is compared with the reference. Simple detectors raise alerts: a single reading far from normal, a sustained drift (EWMA), an accumulating shift (CUSUM) and, where controller outputs are recorded, a control-loop check (pH or temperature drifting while its pump or cooling does not respond). How tight the limits are is set by a false-alarm budget: at most 5% of known-normal batches would have alerted. Every alert shows what changed, by how much, and when.
4. **Suggest explanations.** A reviewable rule book (`knowledge/catalogue.yaml`) links patterns of evidence to explanations: contamination, slow growth, a raw-material/condition change, a sensor fault, or an equipment fault (for example a pump or cooling supply that stops responding, seen from the controller outputs when they are recorded). Explanations are ranked by a transparent score. When the evidence is thin, the answer is *"unresolved, too early to tell"*.
5. **Recommend the next check.** It suggests the quickest, cheapest check that best separates the leading explanations, and keeps a contamination check on the list until contamination is ruled out.
6. **Trace possible sources.** It asks what contaminated batches had in common (medium lot, inoculum lot, bioreactor) compared with clean batches.
7. **Learn from confirmed results.** Recording a test result (e.g. a positive microscopy result) updates the assessment. Only an independent test can mark contamination as *confirmed*.
        """
    )
    st.subheader("Limits")
    st.markdown(
        f"""
- Built and evaluated on **synthetic** α-amylase batches and on external data with no contamination (IndPenSim, a real erythromycin plant); real contamination-detection performance is unknown until tested on MaverickX data.
- On realistic batch-to-batch variation (simulated contaminants added to IndPenSim batches), a contaminant is caught reliably once it changes respiration (CO₂ output, oxygen uptake) by about 40%, and mostly missed at 20% or less.
- The rule book is a **draft** ({kb.review_status}).
- Anomaly scores are not contamination probabilities, and no organism is identified from process data.
- Source tracing only knows the recorded genealogy (medium lot, inoculum lot, bioreactor).
        """
    )


{
    "Campaign overview": page_overview,
    "Batch investigation": page_batch,
    "Detection timing": page_timing,
    "Source tracing": page_tracing,
    "Value estimate": page_value,
    "Load your data": page_load,
    "How it works": page_about,
}[page]()
