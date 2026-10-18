"""Campaign-level pipeline used by the dashboard: load data, then detect, diagnose, trace.

A *campaign* is a folder with ``measurements.csv`` and ``batches.csv``, plus
optional ``investigations.csv`` (recorded check results), ``outcomes.csv`` (known
answers, used only for evaluation and never by the model) and ``profile.yaml`` (a
process profile adapting the rule book's roles to this process's variables).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .diagnosis import DiagnosisResult, Knowledge, diagnose, load_knowledge
from .ingest import (
    IngestResult,
    load_batches,
    load_investigations,
    load_measurements,
    load_outcomes,
)
from .monitor import DEFAULT_CONFIG, MonitorConfig, fit_reference
from .reference import ReferenceProfile
from .schema import INVESTIGATION_COLUMNS
from .tracing import TraceResult, apply_to_reports, batch_statuses, trace_sources


@dataclass
class Campaign:
    ingest: IngestResult
    batches: pd.DataFrame
    investigations: pd.DataFrame
    outcomes: pd.DataFrame | None
    reference_ids: list[str]
    monitored_ids: list[str]
    kb: Knowledge

    @property
    def measurements(self) -> pd.DataFrame:
        return self.ingest.measurements


def load_campaign(
    data_dir: str | Path, n_reference: int = 30, reference_ids: list[str] | None = None
) -> Campaign:
    """Load a campaign. ``reference_ids`` are the batches approved as normal; without
    them, the earliest ``n_reference`` batches are used (the synthetic demo's rule).
    """
    d = Path(data_dir)
    batches = load_batches(d / "batches.csv")
    ingest = load_measurements(
        d / "measurements.csv", batches.set_index("batch_id")["start_time"].to_dict()
    )
    inv_path, out_path = d / "investigations.csv", d / "outcomes.csv"
    investigations = (
        load_investigations(inv_path)
        if inv_path.exists()
        else pd.DataFrame(columns=INVESTIGATION_COLUMNS)
    )
    outcomes = load_outcomes(out_path) if out_path.exists() else None
    ordered = batches.sort_values("start_time")["batch_id"].tolist()
    if reference_ids is None:
        reference_ids = ordered[:n_reference]
    unknown = set(reference_ids) - set(ordered)
    if unknown:
        raise ValueError(f"reference batches not in campaign: {sorted(unknown)}")
    monitored = [b for b in ordered if b not in set(reference_ids)]
    profile = d / "profile.yaml"
    kb = load_knowledge(profile=profile) if profile.exists() else load_knowledge()
    return Campaign(ingest, batches, investigations, outcomes, list(reference_ids), monitored, kb)


def fit(campaign: Campaign, config: MonitorConfig = DEFAULT_CONFIG) -> ReferenceProfile:
    return fit_reference(campaign.measurements, campaign.reference_ids, config)


@dataclass
class CampaignResult:
    diagnosis: DiagnosisResult
    statuses: pd.DataFrame
    trace: TraceResult

    def overview(self, campaign: Campaign) -> pd.DataFrame:
        """One row per monitored batch: what a scientist sees on the overview page."""
        summary = self.diagnosis.monitor_result.summary.set_index("batch_id")
        status = self.statuses.set_index("batch_id")
        start = campaign.batches.set_index("batch_id")["start_time"]
        rows = []
        for b in campaign.monitored_ids:
            r = self.diagnosis.reports.get(b)
            s = summary.loc[b] if b in summary.index else None
            rows.append(
                {
                    "batch_id": b,
                    "start": start.get(b),
                    "status": status["status"].get(b, ""),
                    "process_deviation": s["process_deviation"] if s is not None else "no data",
                    "first_alert_h": s["first_alert_raised_h"] if s is not None else None,
                    "variables_flagged": s["variables_flagged"] if s is not None else "",
                    "leading_explanation": r.assessment["leading_explanation"] if r else "",
                    "contamination_evidence": r.assessment["contamination_evidence"] if r else "",
                    "confidence": r.assessment["confidence"] if r else "",
                    "affected_scope": r.assessment["affected_scope"] if r else "",
                    "next_check": r.next_checks[0]["name"] if r and r.next_checks else "",
                }
            )
        return pd.DataFrame(rows)


def run_campaign(
    campaign: Campaign,
    reference: ReferenceProfile,
    investigations: pd.DataFrame | None = None,
    config: MonitorConfig = DEFAULT_CONFIG,
) -> CampaignResult:
    inv = campaign.investigations if investigations is None else investigations
    m = campaign.measurements
    diagnosis = diagnose(
        m[m["batch_id"].isin(campaign.monitored_ids)], reference, config, inv, kb=campaign.kb
    )
    statuses = batch_statuses(diagnosis, campaign.monitored_ids, campaign.reference_ids)
    trace = trace_sources(campaign.batches, statuses, kb=campaign.kb)
    apply_to_reports(diagnosis, trace)
    return CampaignResult(diagnosis, statuses, trace)


def diagnose_batch_as_of(
    campaign: Campaign,
    reference: ReferenceProfile,
    batch_id: str,
    as_of_h: float | None,
    investigations: pd.DataFrame,
    config: MonitorConfig = DEFAULT_CONFIG,
) -> DiagnosisResult:
    """Replay one batch: only results available by ``as_of_h`` are used."""
    m = campaign.measurements
    return diagnose(
        m[m["batch_id"] == batch_id],
        reference,
        config,
        investigations,
        as_of_h=as_of_h,
        kb=campaign.kb,
    )
