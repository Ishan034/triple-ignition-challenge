"""Contamination investigation dashboard (Streamlit).

    streamlit run app/dashboard.py

Pages: campaign overview (what needs attention, review queue, quick actions), batch
investigation (Summary / Evidence / Actions, with time replay), detection timing, source
tracing, a value estimate from the plant's own figures, loading a plant's own data (a
guided column mapper), and an about page.
"""

from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from dataclasses import replace
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
from contamination import records
from contamination.diagnosis import NOT_MEASURED, load_knowledge
from contamination.evaluate import per_batch
from contamination.importer import (
    FILE_KINDS,
    GENEALOGY_COLUMNS,
    convert,
    convert_mapping,
    draft_mapping,
    mapped_variables,
    mapping_yaml,
    readiness,
    suggest_category,
    suggest_file_kind,
    suggest_roles,
    suggest_variable,
)
from contamination.monitor import DEFAULT_CONFIG
from contamination.pipeline import (
    diagnose_batch_as_of,
    diagnosis_timeline,
    first_named,
    fit_cached,
    load_campaign,
    measurements_path,
    run_campaign,
)
from contamination.plant import load_plant
from contamination.schema import INVESTIGATION_COLUMNS, OUTCOME_CATEGORIES, VARIABLES
from contamination.triage import at_risk, escalation_text, headline, triage
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
FEEDBACK_COLUMNS = records.FEEDBACK_COLUMNS
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.20)
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
def get_campaign(data_dir: str, n_reference: int, budget: float):
    campaign = load_campaign(data_dir, n_reference)
    config = config_for(budget)
    return campaign, fit_cached(campaign, config, records.records_dir(data_dir))


def config_for(budget: float):
    return replace(DEFAULT_CONFIG, false_alarm_budget=budget)


@st.cache_resource(show_spinner="Detecting, diagnosing and tracing sources…")
def get_result(data_dir: str, n_reference: int, budget: float, investigations_csv: str):
    campaign, reference = get_campaign(data_dir, n_reference, budget)
    return run_campaign(campaign, reference, _inv_from_csv(investigations_csv), config_for(budget))


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
def get_timeline(
    data_dir: str, n_reference: int, budget: float, batch_id: str, investigations_csv: str
):
    campaign, reference = get_campaign(data_dir, n_reference, budget)
    return diagnosis_timeline(
        campaign, reference, batch_id, _inv_from_csv(investigations_csv), config_for(budget)
    )


def record_check(row: pd.DataFrame) -> None:
    """Save a check result for this campaign and add it to the session."""
    saved = pd.concat([records.load_checks(data_dir), row], ignore_index=True)
    records.save_checks(data_dir, saved)
    st.session_state["investigations_csv"] = pd.concat(
        [investigations(), row], ignore_index=True
    ).to_csv(index=False)


def record_feedback(row: pd.DataFrame) -> None:
    saved = pd.concat([records.load_feedback(data_dir), row], ignore_index=True)
    records.save_feedback(data_dir, saved)
    st.session_state["feedback_csv"] = records.load_feedback(data_dir).to_csv(index=False)


def records_panel() -> None:
    """Sidebar: what has been recorded for this campaign, download and restore."""
    checks, fb = records.load_checks(data_dir), records.load_feedback(data_dir)
    with st.expander(f"Saved records: {len(checks)} check results, {len(fb)} alert verdicts"):
        st.caption(
            "Check results and alert feedback are saved for this dataset and reloaded next "
            "time. On a hosted app the server's disk is wiped when the app restarts: download "
            "them, and upload the file here to restore."
        )
        c1, c2 = st.columns(2)
        c1.download_button(
            "Checks CSV", checks.to_csv(index=False), "recorded_checks.csv", "text/csv"
        )
        c2.download_button("Feedback CSV", fb.to_csv(index=False), "alert_feedback.csv", "text/csv")
        upload = st.file_uploader("Restore from a downloaded CSV", type=["csv"], key="restore")
        if upload is not None and st.button("Restore"):
            df = pd.read_csv(upload, dtype={"note": "object"})
            try:
                if "verdict" in df.columns:
                    records.save_feedback(
                        data_dir, records.merge_upload(fb, df, records.FEEDBACK_COLUMNS)
                    )
                else:
                    records.save_checks(
                        data_dir, records.merge_upload(checks, df, INVESTIGATION_COLUMNS)
                    )
            except ValueError as e:
                st.error(str(e))
                return
            st.session_state.pop("loaded_from", None)  # reload the session from disk
            st.rerun()


# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("🧫 Contamination investigation")
    choices = {k: v for k, v in DATASETS.items() if measurements_path(v)}
    for folder in sorted(IMPORT_ROOT.glob("*")) if IMPORT_ROOT.exists() else []:
        if measurements_path(folder):
            choices[f"Imported: {folder.name}"] = str(folder)
    wanted = st.session_state.get("data_dir")
    labels = list(choices) + [OTHER]
    index = next((i for i, k in enumerate(labels) if wanted and choices.get(k) == wanted), 0)
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
    budget = st.select_slider(
        "False-alarm budget",
        options=BUDGETS,
        value=DEFAULT_CONFIG.false_alarm_budget,
        format_func=lambda b: f"{b:.0%}",
        help="At most this share of the approved normal batches would have raised an alert. "
        "Lower: fewer false alarms, later or missed detections. Higher: earlier detection, more "
        "batches to review. A new setting recalibrates the limits (up to a few minutes on a "
        "large dataset, once).",
    )
    campaign, reference = get_campaign(data_dir, int(n_reference), budget)
    if (
        "investigations_csv" not in st.session_state
        or st.session_state.get("loaded_from") != data_dir
    ):
        st.session_state["investigations_csv"] = pd.concat(
            [campaign.investigations, records.load_checks(data_dir)], ignore_index=True
        ).to_csv(index=False)
        st.session_state["feedback_csv"] = records.load_feedback(data_dir).to_csv(index=False)
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
        key="page",
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
        f"({campaign.reference_source}) · {len(campaign.monitored_ids)} monitored · "
        f"false-alarm budget {budget:.0%}"
    )
    records_panel()

kb = campaign.kb
plant = load_plant(Path(data_dir) / "profile.yaml")
result = get_result(data_dir, int(n_reference), budget, st.session_state["investigations_csv"])
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


PRIORITY_COLOURS = {
    "act now": "background-color: rgba(235, 104, 52, 0.28)",
    "review": "background-color: rgba(236, 131, 90, 0.13)",
    "at risk": "background-color: rgba(42, 120, 214, 0.15)",
    "no action": "",
}
PRIORITY_ICONS = {"act now": "🔴", "review": "🟠", "at risk": "🔵", "no action": "⚪"}


def actions() -> pd.DataFrame:
    return records.load_actions(data_dir)


def log_actions(rows: list[dict]) -> None:
    records.add_actions(data_dir, rows)


def _open_batch(batch_id: str) -> None:
    """Button callback: jump to the batch page (runs before the rerun, so it may set widgets)."""
    st.session_state["page"] = "Batch investigation"
    st.session_state["batch_pick"] = batch_id


def _next_check(batch_id: str) -> str:
    r = result.diagnosis.reports.get(batch_id)
    return r.next_checks[0]["name"] if r and r.next_checks else ""


def campaign_triage() -> pd.DataFrame:
    return triage(result.overview(campaign), result.trace, actions())


def open_actions_panel(batch_id: str | None = None) -> None:
    """Open actions (all, or one batch's) with a control to mark one done."""
    acts = actions()
    if batch_id is not None:
        acts = acts[acts["batch_id"] == batch_id]
    if acts.empty:
        st.caption("No actions logged yet.")
        return
    st.dataframe(acts, hide_index=True, width="stretch")
    open_ = acts[acts["status"] == "open"]
    if len(open_):
        c1, c2 = st.columns([3, 1])
        pick = c1.selectbox(
            "Open action",
            open_["action_id"].tolist(),
            format_func=lambda i: "#{} {} · {} · {}".format(
                i, *open_.set_index("action_id").loc[i, ["batch_id", "action", "detail"]]
            ),
            key=f"done_pick_{batch_id}",
        )
        if c2.button("Mark done", key=f"done_{batch_id}"):
            records.set_action_status(data_dir, pick, "done")
            st.rerun()


def page_overview():
    st.header("Campaign overview")
    t = campaign_triage()
    n, breakdown = headline(t)
    acts = actions()
    n_open = int((acts["status"] == "open").sum()) if len(acts) else 0
    open_text = f" · {n_open} open action{'s' if n_open != 1 else ''}" if n_open else ""
    if n:
        verb = "needs" if n == 1 else "need"
        (st.error if (t["priority"] == "act now").any() else st.warning)(
            f"**{n} batch{'es' if n != 1 else ''} {verb} attention**: {breakdown}{open_text}.",
            icon="🚨" if (t["priority"] == "act now").any() else "⚠️",
        )
    else:
        st.success(
            "No batch needs attention: every monitored batch stayed within the reference profile."
        )

    ov = t
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
        "Strong source leads", sum(x.tier == "strong lead" for x in result.trace.candidates)
    )

    urgent = t[t["priority"] == "act now"]
    if len(urgent):
        st.subheader("Act now")
        for r in urgent.head(6).itertuples(index=False):
            nxt = _next_check(r.batch_id)
            st.markdown(
                f"🔴 **{r.batch_id}**: {r.status}; leading explanation *{r.leading_explanation}*"
                + (f"; next check: **{nxt}**" if nxt else "")
            )
        if len(urgent) > 6:
            st.caption(f"…and {len(urgent) - 6} more in the table below.")
    risk = t[t["priority"] == "at risk"]
    if len(risk):
        st.markdown(
            f"🔵 **{len(risk)} batch{'es' if len(risk) != 1 else ''} at risk** through a shared "
            "possible source (no alert yet): "
            + ", ".join(risk["batch_id"].head(10))
            + ("…" if len(risk) > 10 else "")
        )

    st.subheader("Review queue")
    only = st.toggle("Only batches that need attention", value=True)
    table = truth_columns(t[t["priority"] != "no action"] if only else t)
    table = table.assign(priority=table["priority"].map(lambda p: f"{PRIORITY_ICONS[p]} {p}"))
    cols = [
        "batch_id", "priority", "why", "contamination_evidence", "confidence",
        "first_alert_h", "next_check", "open_actions", "start", "status",
        "process_deviation", "leading_explanation", "variables_flagged", "affected_scope",
    ]  # fmt: skip
    cols += [c for c in table.columns if c not in cols]
    shown = table[cols]
    st.dataframe(
        shown.style.apply(
            lambda row: [PRIORITY_COLOURS[row["priority"].split(" ", 1)[1]]] * len(row), axis=1
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "start": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
            "first_alert_h": st.column_config.NumberColumn("first alert (h)", format="%.1f"),
        },
    )
    st.caption(
        "Priority orders the review queue; it is not a probability. Contamination evidence and "
        "process deviation are separate: a batch can be strongly abnormal with no contamination "
        "confirmed. *At risk* means the batch used a resource that source tracing ranks as a lead."
    )

    attention = t.loc[t["priority"] != "no action", "batch_id"].tolist()
    if attention:
        st.subheader("Quick actions")
        q1, q2, q3, q4 = st.columns([2, 1, 1, 1])
        b = q1.selectbox(
            "Batch",
            attention,
            format_func=lambda x: (
                f"{PRIORITY_ICONS[t.set_index('batch_id').loc[x, 'priority']]} {x}"
            ),
            key="quick_batch",
        )
        q2.button("Open batch", on_click=_open_batch, args=(b,), width="stretch")
        nxt = _next_check(b)
        if q3.button("Request next check", disabled=not nxt, width="stretch"):
            log_actions([{"batch_id": b, "action": "request check", "detail": nxt}])
            st.toast(f"Check requested for {b}: {nxt}")
            st.rerun()
        if q4.button("Escalate", width="stretch"):
            log_actions([{"batch_id": b, "action": "escalate", "detail": "from overview"}])
            st.toast(f"{b} escalated: open the batch to copy the summary")
            st.rerun()
        if len(risk):
            already = (
                set(acts.loc[acts["action"] == "flag at risk", "batch_id"]) if len(acts) else set()
            )
            todo = [x for x in risk["batch_id"] if x not in already]
            if todo and st.button(f"Flag all {len(todo)} at-risk batches for checking"):
                why = risk.set_index("batch_id")["why"]
                log_actions(
                    [{"batch_id": x, "action": "flag at risk", "detail": why[x]} for x in todo]
                )
                st.rerun()

    with st.expander(f"Actions log ({n_open} open)", expanded=bool(n_open)):
        open_actions_panel()
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
    if st.session_state.get("batch_pick") not in options:
        st.session_state.pop("batch_pick", None)
    left, right = st.columns([1, 2])
    batch_id = left.selectbox(
        "Batch",
        options,
        format_func=lambda b: f"{b}{' ⚑' if b in flagged else ''}",
        key="batch_pick",
    )
    m = campaign.measurements[campaign.measurements["batch_id"] == batch_id]
    max_h = float(np.ceil(m["available_h"].max()))
    replay = right.toggle("Replay: view as of a point in the batch", value=False)
    as_of = right.slider("As of elapsed hour", 0.0, max_h, max_h, 0.5) if replay else None

    inv = investigations()
    diag = diagnose_batch_as_of(campaign, reference, batch_id, as_of, inv, config_for(budget))
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

    if report is None:
        st.success(
            "No alerts"
            + (
                f" as of {as_of:g} h."
                if as_of is not None
                else ": the batch stayed within the reference profile."
            )
        )
        risk = at_risk(result.trace).get(batch_id)
        if risk:
            st.info(
                f"🔵 At risk: this batch used {', '.join(risk)}, which source tracing ranks as "
                "a possible source. Consider a check even without an alert.",
            )
        st.plotly_chart(
            batch_figure(batch_id, scores, alerts, as_of), width="stretch", theme="streamlit"
        )
        return

    summary, evidence, act = st.tabs(["Summary", "Evidence", "Actions"])
    with summary:
        summary_section(batch_id, report, scores, alerts, as_of)
    with evidence:
        evidence_section(batch_id, report, alerts)
    with act:
        actions_section(batch_id, report, alerts, as_of, max_h, inv)


def summary_section(batch_id, report, scores, alerts, as_of):
    a = report.assessment
    cards = [
        ("Leading explanation", a["leading_explanation"]),
        ("Contamination evidence", a["contamination_evidence"]),
        ("Process deviation", a["process_deviation"]),
        ("Confidence", a["confidence"]),
    ]
    for col, (label, value) in zip(st.columns(4), cards):
        with col.container(border=True):
            st.caption(label)
            st.markdown(f"**{value}**")
    if report.next_checks:
        ch = report.next_checks[0]
        st.info(
            f"**Next check: {ch['name']}** (~{ch['turnaround_h']:g} h, {ch['relative_cost']} "
            f"cost). {ch['reason']}",
            icon="🔬",
        )
    if plant and plant.wording.get("today_detection"):
        st.caption(f"Today's method at this plant: {plant.wording['today_detection']}.")
    st.markdown("**What changed**")
    first = sorted(alerts, key=lambda x: x.first_raised_h)
    lines = []
    for al in first[:5]:
        lines.append(
            f"- **{al.variable}** {al.direction} from {al.start_h:g} h: {al.observed_at_peak:.3g} vs "
            f"reference {al.reference_median_at_peak:.3g} ({al.detector}, {al.persistence})"
        )
    if len(first) > 5:
        lines.append(f"- …and {len(first) - 5} more alerts (Evidence tab)")
    st.markdown("\n".join(lines))
    st.markdown(
        f"**Quality impact:** {a['quality_impact']}  \n**Affected scope:** {a['affected_scope']}"
    )
    st.plotly_chart(
        batch_figure(batch_id, scores, alerts, as_of), width="stretch", theme="streamlit"
    )


def evidence_section(batch_id, report, alerts):
    st.subheader("Explanations")
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
    st.dataframe(hyp, hide_index=True, width="stretch")
    st.caption(
        "Score = supporting weights − contradicting weights. It orders explanations for review; it is not a probability."
    )
    st.subheader("Evidence items")
    ev = pd.DataFrame(
        [
            {"id": e.id, "evidence": e.name, "state": e.state, "detail": e.detail}
            for e in report.evidence.values()
        ]
    )
    st.dataframe(ev, hide_index=True, width="stretch")
    st.subheader("Alerts")
    al = pd.DataFrame([x.to_dict() for x in alerts])
    cols = [
        "variable", "detector", "direction", "start_h", "end_h", "first_raised_h", "persistence",
        "observed_at_peak", "reference_median_at_peak", "robust_z_at_peak",
        "corroborating_variables", "detail", "data_quality_warnings",
    ]  # fmt: skip
    st.dataframe(al[cols], hide_index=True, width="stretch")
    st.subheader("Possible sources (leads, not proof)")
    if not report.sources:
        st.write("None: source tracing not run, or contamination not suspected.")
    for src in report.sources:
        st.markdown(
            f"- **{src['resource']}** ({src['tier']}): {src['explanation']}  \n  *Suggested check:* {src['suggested_check']}"
        )
    st.subheader("How it developed")
    timeline_section(batch_id)
    with st.expander("Full report"):
        st.markdown(report.to_markdown())


def actions_section(batch_id, report, alerts, as_of, max_h, inv):
    st.subheader("Request a check")
    if report.next_checks:
        names = [ch["name"] for ch in report.next_checks]
        r1, r2 = st.columns([3, 1])
        pick = r1.selectbox("Check to request", names, key="request_pick")
        if r2.button("Request check", width="stretch"):
            log_actions([{"batch_id": batch_id, "action": "request check", "detail": pick}])
            st.toast(f"Requested: {pick}")
            st.rerun()
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
            max(200.0, max_h + 48.0),  # long batches run past 200 h
            float(as_of if as_of is not None else max_h),
        )
        note = st.text_input("Note")
        if st.form_submit_button("Add result"):
            row = pd.DataFrame(
                [[batch_id, check, res, hour, note or None]], columns=INVESTIGATION_COLUMNS
            )
            record_check(row)
            st.rerun()
    batch_inv = inv[inv["batch_id"] == batch_id]
    if len(batch_inv):
        st.dataframe(batch_inv, hide_index=True, width="stretch")

    st.subheader("Alert feedback")
    feedback_section(batch_id, alerts)

    linked = {
        b: why
        for b, why in at_risk(result.trace, set(campaign.monitored_ids)).items()
        if any(why_r.split(" (")[0] in {s["resource"] for s in report.sources} for why_r in why)
    }
    if linked:
        st.subheader("Batches at risk through the same source")
        st.write(
            "These batches used a resource this batch shares with other cases, and have no "
            "contamination call yet: " + ", ".join(sorted(linked))
        )
        if st.button(f"Flag {len(linked)} at-risk batches for checking"):
            log_actions(
                [
                    {
                        "batch_id": b,
                        "action": "flag at risk",
                        "detail": f"linked to {batch_id}: " + ", ".join(w),
                    }
                    for b, w in sorted(linked.items())
                ]
            )
            st.rerun()

    st.subheader("Escalate")
    text = escalation_text(batch_id, report, plant.name if plant else "", as_of)
    st.text_area("Summary to send (copy it, or download)", text, height=260)
    e1, e2, e3 = st.columns(3)
    e1.download_button("Download summary (.txt)", text, f"{batch_id}_escalation.txt", "text/plain")
    e2.download_button(
        "Download full report (Markdown)",
        report.to_markdown(),
        f"{batch_id}_report.md",
        "text/markdown",
    )
    if e3.button("Log escalation"):
        log_actions(
            [
                {
                    "batch_id": batch_id,
                    "action": "escalate",
                    "detail": report.assessment["leading_explanation"],
                }
            ]
        )
        st.rerun()
    st.caption(
        "Nothing is sent from the dashboard: escalation is logged here and the summary is yours to send."
    )

    st.subheader("Actions for this batch")
    open_actions_panel(batch_id)


def timeline_section(batch_id: str):
    st.caption(
        "The batch replayed hour by hour, using only the results available at each hour: "
        "when the first alert came, and when each explanation first became the leading one."
    )
    tl = get_timeline(
        data_dir, int(n_reference), budget, batch_id, st.session_state["investigations_csv"]
    )
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
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    named = first_named(tl)
    if len(named):
        st.dataframe(
            named.rename(columns={"hour": "first leading at (h)"}),
            hide_index=True,
            width="stretch",
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
        verdict = st.radio("Verdict", list(records.VERDICTS), horizontal=True)
        note = st.text_input("Note (what was found)")
        if st.form_submit_button("Save feedback"):
            a = labels[choice]
            row = pd.DataFrame(
                [[batch_id, a.variable, a.detector, a.start_h, verdict, note or None]],
                columns=FEEDBACK_COLUMNS,
            )
            record_feedback(row)
            st.rerun()
    fb = feedback()
    if len(fb[fb["batch_id"] == batch_id]):
        st.dataframe(fb[fb["batch_id"] == batch_id], hide_index=True, width="stretch")
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
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    st.dataframe(pb, hide_index=True, width="stretch")
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
    sources = {"Demo placeholders": None, "MaverickX draft profile": "maverickx"}
    if plant is not None and plant.value_defaults:
        sources = {f"{plant.name}": Path(data_dir) / "profile.yaml", **sources}
    chosen = sources[st.selectbox("Pre-fill from", list(sources))]
    prof = load_plant(chosen) if chosen else None
    defaults = prof.value_defaults if prof else {}

    def dflt(key: str, placeholder: float) -> float:
        d = defaults.get(key)
        return float(d.value) if d is not None and d.value is not None else placeholder

    if prof is None:
        st.warning(
            "The pre-filled values are placeholders for the demo, not MaverickX figures. "
            "Replace them before using the result.",
            icon="⚠️",
        )
    else:
        unconfirmed = [
            f"- **{k.replace('_', ' ')}**: "
            + (f"{v.value:,g} ({v.source})" if v.value is not None else f"placeholder ({v.source})")
            for k, v in defaults.items()
            if not v.confirmed
        ]
        if unconfirmed:
            st.warning(
                f"From **{prof.name}**. Not yet confirmed by the plant, so the result is only "
                "illustrative until these are replaced:\n" + "\n".join(unconfirmed),
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
        batches = st.number_input(
            "Batches per year", 1.0, 10000.0, dflt("batches_per_year", 100.0), 10.0
        )
        rate = (
            st.number_input(
                "Share lost to contamination today (%)",
                0.0,
                100.0,
                100 * dflt("contamination_rate", 0.05),
                0.5,
            )
            / 100
        )
        lost = st.number_input(
            "Cost of one lost batch", 0.0, 1e8, dflt("lost_batch_cost", 50000.0), 1000.0
        )
        hourly = st.number_input(
            "Cost per hour of running a doomed batch",
            0.0,
            1e6,
            dflt("running_cost_per_hour", 500.0),
            50.0,
        )
        test_cost = st.number_input(
            "Cost of one expensive confirmatory test", 0.0, 1e6, dflt("test_cost", 10000.0), 500.0
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
            "Later batches spared once a shared source is found",
            0.0,
            50.0,
            dflt("later_batches_saved_per_source", 1.0),
            0.5,
        )
        tests = st.number_input(
            "Expensive tests avoided per case", 0.0, 20.0, dflt("tests_avoided_per_case", 1.0), 0.5
        )
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
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"value_per_year": st.column_config.NumberColumn(format="%,.0f")},
    )


def _template_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(TEMPLATE_DIR.glob("*")):
            if f.is_file():
                z.write(f, f.name)
    return buf.getvalue()


PROFILES = {
    "MaverickX (draft profile, to confirm)": "maverickx",
    "Generic (choose roles yourself)": None,
}
NONE = "(none)"
IGNORE = "ignore"
NEW = "new variable…"


@st.cache_data(show_spinner=False)
def _read_upload(name: str, data: bytes) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


def _pick(label, columns, default, key, optional=False):
    options = ([NONE] if optional else []) + list(columns)
    index = options.index(default) if default in options else 0
    value = st.selectbox(label, options, index=index, key=key)
    return None if value == NONE else value


def _variable_editor(names, suggested: dict, key: str) -> tuple[dict, dict, list[str]]:
    """Table of export names -> variables. Returns (name map, new variables, problems)."""
    known = sorted(VARIABLES)
    rows = []
    for n in names:
        v = suggested.get(n)
        rows.append(
            {
                "in your export": str(n),
                "maps to": v if v else IGNORE,
                "new variable name": "",
                "unit": VARIABLES[v].unit if v in VARIABLES else "",
                "noise floor": None,
                "plausible min": None,
                "plausible max": None,
            }
        )
    edited = st.data_editor(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        key=key,
        disabled=["in your export"],
        column_config={
            "maps to": st.column_config.SelectboxColumn(
                options=[IGNORE, *known, NEW], required=True
            ),
            "noise floor": st.column_config.NumberColumn(
                help="New variables only: the smallest spread that counts as real (sensor resolution)."
            ),
        },
    )
    mapping, new, problems = {}, {}, []
    for r in edited.itertuples(index=False):
        target = r[1]
        if target == IGNORE:
            continue
        if target == NEW:
            name = "".join(ch if ch.isalnum() else "_" for ch in str(r[2]).strip().lower()).strip(
                "_"
            )
            nums = (r[4], r[5], r[6])
            if not name or any(x is None or pd.isna(x) for x in nums) or not str(r[3]).strip():
                problems.append(
                    f"'{r[0]}': a new variable needs a name, unit, noise floor and plausible range"
                )
                continue
            new[name] = {
                "unit": str(r[3]).strip(),
                "min_scale": float(r[4]),
                "plausible_range": [float(r[5]), float(r[6])],
                "description": str(r[0]),
            }
            target = name
        mapping[str(r[0])] = target
    return mapping, new, problems


def _file_spec(fname: str, df: pd.DataFrame, export_names) -> tuple[str, dict, dict, list[str]]:
    """Widgets for one uploaded file. Returns (kind, spec, new variables, problems)."""
    st.dataframe(df.head(5), hide_index=True, width="stretch")
    kind = st.selectbox(
        "This file holds",
        FILE_KINDS,
        index=FILE_KINDS.index(suggest_file_kind(df)),
        key=f"kind_{fname}",
    )
    if kind == "ignore":
        return kind, {}, {}, []
    draft = draft_mapping({fname: df}, export_names, kinds={fname: kind})
    cols = list(df.columns)
    new, problems = {}, []
    if kind in FILE_KINDS[:2]:
        d = draft["measurements"][0]
        c1, c2, c3 = st.columns(3)
        with c1:
            batch = _pick("Batch ID column", cols, d.get("batch_id"), f"batch_{fname}")
        time_kind = c2.radio(
            "Time is",
            ["date and time", "hours since batch start"],
            index=0 if "timestamp" in d or "elapsed_hours" not in d else 1,
            key=f"tkind_{fname}",
        )
        with c3:
            tcol = _pick(
                "Time column",
                cols,
                d.get("timestamp") or d.get("elapsed_hours"),
                f"time_{fname}",
            )
        spec = {
            "file": fname,
            "layout": "wide" if kind == FILE_KINDS[0] else "long",
            "batch_id": batch,
        }
        spec["timestamp" if time_kind == "date and time" else "elapsed_hours"] = tcol
        if kind == FILE_KINDS[0]:
            spec["source"] = "online_sensor"
            spec["result_delay_h"] = st.number_input(
                "Result delay (h): how long after the reading it can be seen",
                0.0, 240.0, 0.0, 0.5, key=f"delay_{fname}",
            )  # fmt: skip
            others = [c for c in cols if c not in (batch, tcol)]
            columns, new, problems = _variable_editor(others, d.get("columns", {}), f"vars_{fname}")
            spec["columns"] = columns
            units = {}
            for c, v in columns.items():
                unit = new.get(v, {}).get("unit") or (VARIABLES[v].unit if v in VARIABLES else None)
                if unit:
                    units[v] = unit
            spec["units"] = units
        else:
            spec["source"] = "offline_assay"
            l1, l2, l3, l4 = st.columns(4)
            with l1:
                avail = _pick(
                    "Result time column", cols, d.get("available_at"), f"avail_{fname}", True
                )
            with l2:
                var = _pick("Test / variable name column", cols, d.get("variable"), f"var_{fname}")
            with l3:
                val = _pick("Value column", cols, d.get("value"), f"val_{fname}")
            with l4:
                unit = _pick("Unit column", cols, d.get("unit"), f"unit_{fname}", True)
            spec |= {"variable": var, "value": val}
            if unit:
                spec["unit"] = unit
            if avail:
                spec["available_at"] = avail
            else:
                spec["result_delay_h"] = st.number_input(
                    "No result time: typical delay from sample to result (h)",
                    0.0, 240.0, 0.0, 0.5, key=f"delay_{fname}",
                )  # fmt: skip
            names = df[var].dropna().astype(str).unique()[:500] if var else []
            suggested = {n: suggest_variable(n, export_names) for n in names}
            spec["variable_map"], new, problems = _variable_editor(
                names, suggested, f"vars_{fname}"
            )
        return kind, spec, new, problems
    if kind == "batch records":
        d = draft["batches"]["columns"]
        spec = {"file": fname, "columns": {}}
        fields = [
            ("batch_id", "Batch ID", False),
            ("start_time", "Start (inoculation) time", False),
        ]
        fields += [(g, g.replace("_", " ").capitalize(), True) for g in GENEALOGY_COLUMNS]
        for col, (f, label, optional) in zip(st.columns(len(fields)), fields):
            with col:
                c = _pick(label, cols, d.get(f), f"{f}_{fname}", optional)
            if c:
                spec["columns"][f] = c
        return kind, spec, {}, []
    if kind == "outcomes":
        d = draft["outcomes"]
        spec = {"file": fname, "columns": {}}
        fields = [
            ("batch_id", "Batch ID", False),
            ("category", "Outcome", False),
            ("onset_h", "Problem started (h)", True),
            ("reference_event_h", "First noticed (h)", True),
            ("reason", "Comment", True),
        ]
        for col, (f, label, optional) in zip(st.columns(len(fields)), fields):
            with col:
                c = _pick(label, cols, d["columns"].get(f), f"{f}_{fname}", optional)
            if c:
                spec["columns"][f] = c
        labels = df[spec["columns"]["category"]].dropna().astype(str).unique()[:100]
        guess = {lab: suggest_category(lab) for lab in labels}
        edited = st.data_editor(
            pd.DataFrame(
                {"outcome in your file": labels, "means": [guess[x] or "" for x in labels]}
            ),
            hide_index=True,
            key=f"cats_{fname}",
            disabled=["outcome in your file"],
            column_config={
                "means": st.column_config.SelectboxColumn(options=list(OUTCOME_CATEGORIES))
            },
        )
        cmap = dict(zip(edited["outcome in your file"], edited["means"]))
        unmapped = [k for k, v in cmap.items() if not v]
        spec["category_map"] = cmap
        return kind, spec, {}, [f"outcome '{k}' needs a meaning" for k in unmapped]
    d = draft.get("observations", [{}])[0]
    spec = {"file": fname}
    for col, (f, label) in zip(
        st.columns(3), [("batch_id", "Batch ID"), ("timestamp", "Time"), ("text", "Note text")]
    ):
        with col:
            spec[f] = _pick(label, cols, d.get(f), f"{f}_{fname}")
    return kind, spec, {}, []


def guided_mapper(name: str, profile: str | None) -> None:
    plant_p = load_plant(profile) if profile else None
    export_names = plant_p.export_names if plant_p else None
    files = st.file_uploader(
        "Your exports (CSV or Excel), all at once",
        accept_multiple_files=True,
        type=["csv", "xlsx", "xls"],
        key="mapper_files",
    )
    if not files:
        st.info(
            "Upload the files exported from your historian, lab system and batch records. "
            "Each column gets a suggested meaning; you confirm or change it. No YAML needed."
        )
        return
    frames = {f.name: _read_upload(f.name, f.getvalue()) for f in files}
    m: dict = {"process_name": name, "variables": {}, "measurements": []}
    problems: list[str] = []
    st.markdown("**1. What each file holds and which columns to use**")
    for fname, df in frames.items():
        with st.expander(f"📄 {fname}: {len(df):,} rows, {len(df.columns)} columns", expanded=True):
            kind, spec, new, probs = _file_spec(fname, df, export_names)
        problems += probs
        m["variables"] |= new
        if kind in FILE_KINDS[:2]:
            m["measurements"].append(spec)
        elif kind == "batch records":
            m["batches"] = spec
        elif kind == "outcomes":
            m["outcomes"] = spec
        elif kind == "operator notes":
            m.setdefault("observations", []).append(spec)

    st.markdown("**2. Which signal plays each role in the diagnosis rules**")
    variables = mapped_variables(m)
    roles = suggest_roles(variables)
    if profile:
        for role, v in load_knowledge(profile=profile).roles.items():
            if v in variables:
                roles[role] = v
    chosen = {}
    role_cols = st.columns(4)
    for i, (role, default) in enumerate(roles.items()):
        options = [*variables, NOT_MEASURED]
        chosen[role] = role_cols[i % 4].selectbox(
            role.replace("_", " "),
            options,
            index=options.index(default) if default in options else len(options) - 1,
            key=f"role_{role}",
        )
    if profile and chosen == load_knowledge(profile=profile).roles:
        m["use_profile"] = profile
    else:
        m["roles"] = chosen
        if profile:
            m["linked_signals"] = list(load_knowledge(profile=profile).linked_signals)

    if not m["measurements"]:
        problems.append("no file is marked as online signals or lab results")
    if "batches" not in m:
        problems.append("no file is marked as batch records (needed for start times and genealogy)")
    st.markdown("**3. Convert**")
    for p in problems:
        st.warning(p, icon="⚠️")
    c1, c2 = st.columns(2)
    c2.download_button(
        "Download this mapping (mapping.yaml)",
        mapping_yaml(m),
        "mapping.yaml",
        "text/yaml",
        help="Reuse it next time, or from the command line: python -m contamination.importer convert",
    )
    if c1.button(
        "Convert and check", type="primary", disabled=bool(problems), key="guided_convert"
    ):
        raw = Path(tempfile.mkdtemp(prefix="upload_"))
        for f in files:
            (raw / Path(f.name).name).write_bytes(f.getvalue())
        _convert(lambda out: convert_mapping(m, raw, out), name)


def _convert(run, name: str) -> None:
    out = IMPORT_ROOT / ("".join(ch for ch in name if ch.isalnum() or ch in "-_") or "imported")
    try:
        res = run(out)
    except (ValueError, KeyError, TypeError) as e:
        st.error(f"Could not convert: {e}")
        return
    st.success(
        f"Converted into `{out}`: " + ", ".join(f"{k} {len(v)} rows" for k, v in res.tables.items())
    )
    if res.unmapped:
        with st.expander(f"{len(res.unmapped)} columns or names not imported"):
            st.write(res.unmapped)
    st.session_state["imported_dir"] = str(out)


def yaml_upload(name: str) -> None:
    if TEMPLATE_DIR.exists():
        st.download_button(
            "Download the data template (example exports + mapping.yaml)",
            _template_zip(),
            "data-template.zip",
            "application/zip",
        )
    files = st.file_uploader(
        "Exports and mapping.yaml",
        accept_multiple_files=True,
        type=["csv", "xlsx", "xls", "yaml", "yml"],
        key="yaml_files",
    )
    if files and st.button("Convert and check", key="yaml_convert"):
        if not any(f.name in ("mapping.yaml", "mapping.yml") for f in files):
            st.error("Include a mapping.yaml (see the template).")
            return
        raw = Path(tempfile.mkdtemp(prefix="upload_"))
        for f in files:
            (raw / Path(f.name).name).write_bytes(f.getvalue())
        mapping = raw / ("mapping.yaml" if (raw / "mapping.yaml").exists() else "mapping.yml")
        _convert(lambda out: convert(mapping, out), name)


def page_load():
    st.header("Load your data")
    st.write(
        "Upload your plant's exports (CSV or Excel). The data is converted, checked, and can then "
        "be monitored here. Values are copied as they are: nothing is filled, rescaled or deleted."
    )
    c1, c2 = st.columns(2)
    name = c1.text_input("Name for this dataset", "maverickx")
    profile = PROFILES[c2.selectbox("Process profile", list(PROFILES))]
    if profile:
        p = load_plant(profile)
        if p:
            st.caption(f"Profile: {p.name}. {p.status}.")
    guided, from_yaml = st.tabs(["Guided column mapping", "I have a mapping.yaml"])
    with guided:
        guided_mapper(name, profile)
    with from_yaml:
        yaml_upload(name)
    target = st.session_state.get("imported_dir")
    if target:
        st.subheader("Readiness")
        r = readiness(target)
        icon = {"ok": "✅", "warn": "⚠️", "missing": "❌"}
        r.insert(0, "", r["status"].map(icon))
        st.dataframe(r, hide_index=True, width="stretch")
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
    st.plotly_chart(genealogy_figure(trace), width="stretch", theme="streamlit")
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
        width="stretch",
        column_config={
            "attack_rate": st.column_config.NumberColumn(format="percent"),
            "attack_rate_elsewhere": st.column_config.NumberColumn(format="percent"),
            "p_value": st.column_config.NumberColumn(format="%.2g"),
            "p_adjusted": st.column_config.NumberColumn(format="%.2g"),
        },
    )
    with st.expander("Batch statuses used for tracing"):
        st.dataframe(truth_columns(trace.statuses), hide_index=True, width="stretch")


def page_about():
    st.header("How it works")
    st.markdown(
        """
The system helps scientists investigate contamination faster. **It supports decisions; it does not make them.**

1. **Check the data.** Readings that are missing, implausible, in the wrong unit or duplicated are flagged, never silently deleted or filled with zero.
2. **Learn what normal looks like.** From batches scientists approve as normal, it learns the typical value of each measurement at each hour, and how much it usually varies. This *reference profile* is then frozen and versioned.
3. **Spot unusual behaviour.** Each new reading is compared with the reference. Simple detectors raise alerts: a single reading far from normal, a sustained drift (EWMA), an accumulating shift (CUSUM), a linked-signal check (at least two linked signals such as CO₂ output and dissolved oxygen changing together from the batch's own recent past) and, where controller outputs are recorded, a control-loop check (pH or temperature drifting while its pump or cooling does not respond). How tight the limits are is set by a false-alarm budget: at most 5% of known-normal batches would have alerted. Every alert shows what changed, by how much, and when.
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
- On realistic batch-to-batch variation (simulated contaminants added to IndPenSim batches), a contaminant that changes respiration (CO₂ output, oxygen uptake) by about 20% is caught in 27 of 30 batches within about 19 h; at 10% it is missed.
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
