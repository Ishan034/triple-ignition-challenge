"""Structured alert objects.

An alert reports a process deviation and the evidence behind it. It deliberately has
no "contamination probability" field. ``contamination_confirmed`` stays False until
independent evidence (for example a confirmatory microbiological test) is recorded
by a scientist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

INTERPRETATION = (
    "Process deviation flag for scientist review. Not a contamination probability and "
    "not an identification of any organism."
)


@dataclass
class Alert:
    batch_id: str
    variable: str
    detector: str
    stage: str
    start_h: float  # measurement time of the first point over the limit
    end_h: float  # measurement time of the last consecutive point over the limit
    first_raised_h: float  # when the alert could first have been raised (result availability)
    n_points: int
    persistence: str  # "isolated" (one reading) or "persistent"
    direction: str  # "high" or "low" relative to the reference
    peak_statistic: float
    limit: float
    observed_at_peak: float
    reference_median_at_peak: float
    reference_scale_at_peak: float
    robust_z_at_peak: float
    algorithm_version: str
    reference_version: str
    corroborating_variables: list[str] = field(default_factory=list)
    data_quality_warnings: list[str] = field(default_factory=list)
    detail: str = ""  # e.g. which linked signals moved together
    contamination_confirmed: bool = False
    interpretation: str = INTERPRETATION

    @property
    def corroborated(self) -> bool:
        return bool(self.corroborating_variables)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["corroborating_variables"] = ", ".join(self.corroborating_variables)
        d["data_quality_warnings"] = "; ".join(self.data_quality_warnings)
        return d

    def explain(self) -> str:
        return (
            f"{self.batch_id} {self.variable} ({self.detector}): {self.direction} deviation "
            f"from {self.start_h:g} h to {self.end_h:g} h ({self.persistence}, {self.n_points} pts). "
            f"Peak: observed {self.observed_at_peak:.3g} vs reference {self.reference_median_at_peak:.3g} "
            f"(robust z {self.robust_z_at_peak:+.1f}; statistic {self.peak_statistic:+.2f}, "
            f"limit {self.limit:.2f}). Corroborating variables: "
            f"{', '.join(self.corroborating_variables) or 'none'}."
        )
