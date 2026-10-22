"""Campaign-level pipeline used by the dashboard: load data, then detect, diagnose, trace.

A *campaign* is a folder with ``measurements.csv`` (or ``measurements.csv.gz``) and
``batches.csv``, plus ``reference_batches.txt`` (the batches approved as normal, one
per line) when the reference is not simply the earliest batches, and
optional ``investigations.csv`` (recorded check results), ``outcomes.csv`` (known
answers, used only for evaluation and never by the model), ``profile.yaml`` (a
process profile adapting the rule book's roles to this process's variables) and
``variables.yaml`` (process-specific variables, written by ``contamination.importer``).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .diagnosis import DiagnosisResult, Knowledge, diagnose, load_knowledge
from .importer import load_variables
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
    data_dir: Path | None = None
    reference_source: str = "earliest batches"

    @property
    def measurements(self) -> pd.DataFrame:
        return self.ingest.measurements


REFERENCE_LIST = "reference_batches.txt"
REFERENCE_CACHE = "reference_profile.json"


def measurements_path(data_dir: str | Path) -> Path | None:
    """The campaign's measurements file, plain or gzip-compressed."""
    for name in ("measurements.csv", "measurements.csv.gz"):
        if (Path(data_dir) / name).exists():
            return Path(data_dir) / name
    return None


def load_campaign(
    data_dir: str | Path, n_reference: int = 30, reference_ids: list[str] | None = None
) -> Campaign:
    """Load a campaign. ``reference_ids`` are the batches approved as normal; without
    them, the list in ``reference_batches.txt`` is used if the folder has one, otherwise
    the earliest ``n_reference`` batches (the synthetic demo's rule).
    """
    d = Path(data_dir)
    load_variables(d)
    batches = load_batches(d / "batches.csv")
    m_path = measurements_path(d)
    if m_path is None:
        raise FileNotFoundError(f"no measurements.csv(.gz) in {d}")
    ingest = load_measurements(m_path, batches.set_index("batch_id")["start_time"].to_dict())
    inv_path, out_path = d / "investigations.csv", d / "outcomes.csv"
    investigations = (
        load_investigations(inv_path)
        if inv_path.exists()
        else pd.DataFrame(columns=INVESTIGATION_COLUMNS)
    )
    outcomes = load_outcomes(out_path) if out_path.exists() else None
    ordered = batches.sort_values("start_time")["batch_id"].tolist()
    source = "given list"
    if reference_ids is None and (d / REFERENCE_LIST).exists():
        reference_ids = [b for b in (d / REFERENCE_LIST).read_text().split() if b]
        source = f"{REFERENCE_LIST} ({len(reference_ids)} approved batches)"
    if reference_ids is None:
        reference_ids = ordered[:n_reference]
        source = f"earliest {len(reference_ids)} batches"
    unknown = set(reference_ids) - set(ordered)
    if unknown:
        raise ValueError(f"reference batches not in campaign: {sorted(unknown)}")
    monitored = [b for b in ordered if b not in set(reference_ids)]
    profile = d / "profile.yaml"
    kb = load_knowledge(profile=profile) if profile.exists() else load_knowledge()
    return Campaign(
        ingest, batches, investigations, outcomes, list(reference_ids), monitored, kb, d, source
    )


def fit(campaign: Campaign, config: MonitorConfig = DEFAULT_CONFIG) -> ReferenceProfile:
    return fit_reference(campaign.measurements, campaign.reference_ids, config)


def _fingerprint(campaign: Campaign, config: MonitorConfig) -> str:
    h = hashlib.sha256(measurements_path(campaign.data_dir).read_bytes())
    h.update(repr((sorted(campaign.reference_ids), asdict(config))).encode())
    return h.hexdigest()[:16]


def fit_cached(campaign: Campaign, config: MonitorConfig = DEFAULT_CONFIG) -> ReferenceProfile:
    """Fit, or reuse ``reference_profile.json`` in the campaign folder when it was fitted
    on the same measurements, approved batches and settings (calibration on hundreds of
    batches takes minutes)."""
    if campaign.data_dir is None:
        return fit(campaign, config)
    key, path = _fingerprint(campaign, config), campaign.data_dir / REFERENCE_CACHE
    if path.exists():
        try:
            cached = ReferenceProfile.load(path)
            if cached.calibration.get("fingerprint") == key:
                return cached
        except (ValueError, KeyError):
            pass
    reference = fit(campaign, config)
    reference.calibration["fingerprint"] = key
    try:
        reference.save(path)
    except OSError:
        pass  # read-only folder: just don't cache
    return reference


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


def diagnosis_timeline(
    campaign: Campaign,
    reference: ReferenceProfile,
    batch_id: str,
    investigations: pd.DataFrame,
    config: MonitorConfig = DEFAULT_CONFIG,
    max_points: int = 40,
) -> pd.DataFrame:
    """How the assessment of one batch developed: replayed at regular hours from its
    first alert to the end, using only what was available at each hour."""
    m = campaign.measurements
    m = m[m["batch_id"] == batch_id]
    full = diagnose(m, reference, config, investigations, kb=campaign.kb).reports.get(batch_id)
    if full is None or not full.alerts:
        return pd.DataFrame(columns=["hour", "leading_id", "leading_explanation", "alerts"])
    start = min(a.first_raised_h for a in full.alerts)
    end = float(m["available_h"].max())
    step = max(1.0, round((end - start) / max_points))
    rows = []
    for h in np.arange(start, end + step, step):
        r = diagnose(m, reference, config, investigations, as_of_h=float(h), kb=campaign.kb)
        rep = r.reports.get(batch_id)
        lead = rep.leading if rep else None
        rows.append(
            {
                "hour": float(h),
                "leading_id": lead.id if lead else "",
                "leading_explanation": lead.name if lead else "no alerts yet",
                "alerts": len(rep.alerts) if rep else 0,
            }
        )
    return pd.DataFrame(rows)


def first_named(timeline: pd.DataFrame) -> pd.DataFrame:
    """The first replay hour at which each explanation became the leading one."""
    named = timeline[~timeline["leading_id"].isin(("", "H05"))]
    return named.groupby(["leading_id", "leading_explanation"], as_index=False)["hour"].min()
