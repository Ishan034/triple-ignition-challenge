"""Rough yearly value of earlier detection and source tracing, from the plant's own figures.

This is arithmetic on numbers the user enters, not a prediction. The model's own
contribution is limited to what evaluation measured (how many hours earlier it warns,
how many false alarms it raises); every cost and rate must come from the plant.
Results on synthetic data must not be presented as real savings.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ValueInputs:
    batches_per_year: float
    contamination_rate: float  # share of batches lost to contamination today
    lost_batch_cost: float  # materials, vessel time and labour of one lost batch
    running_cost_per_hour: float  # cost of keeping a doomed batch running for an hour
    hours_earlier: float  # how much earlier the model warns than today's method
    share_from_shared_source: float  # share of cases that come from a shared source
    later_batches_saved_per_source: float  # batches spared once a shared source is found
    tests_avoided_per_case: float  # expensive tests replaced by a targeted cheap check
    test_cost: float
    good_batches_saved_per_year: float  # batches not scrapped because a sensor fault was recognised
    false_alarm_share: float  # share of normal batches that raise an alert needing a check
    cost_per_false_alarm: float  # staff time and a quick check to dismiss one


def estimate(v: ValueInputs) -> pd.DataFrame:
    """Yearly value by mechanism (currency of the inputs). Negative rows are costs."""
    cases = v.batches_per_year * v.contamination_rate
    rows = [
        (
            "Later batches spared by finding a shared source",
            cases
            * v.share_from_shared_source
            * v.later_batches_saved_per_source
            * v.lost_batch_cost,
            (
                f"{cases:.1f} cases x {v.share_from_shared_source:.0%} from a shared source x "
                f"{v.later_batches_saved_per_source:g} batches spared x {v.lost_batch_cost:,.0f}"
            ),
        ),
        (
            "Doomed batches stopped earlier",
            cases * v.hours_earlier * v.running_cost_per_hour,
            f"{cases:.1f} cases x {v.hours_earlier:g} h earlier x {v.running_cost_per_hour:,.0f} per hour",
        ),
        (
            "Expensive tests replaced by a targeted check",
            cases * v.tests_avoided_per_case * v.test_cost,
            f"{cases:.1f} cases x {v.tests_avoided_per_case:g} tests x {v.test_cost:,.0f}",
        ),
        (
            "Good batches not scrapped (sensor fault recognised)",
            v.good_batches_saved_per_year * v.lost_batch_cost,
            f"{v.good_batches_saved_per_year:g} batches x {v.lost_batch_cost:,.0f}",
        ),
        (
            "Cost of checking false alarms",
            -v.batches_per_year
            * (1 - v.contamination_rate)
            * v.false_alarm_share
            * v.cost_per_false_alarm,
            (
                f"{v.batches_per_year * (1 - v.contamination_rate):.0f} normal batches x "
                f"{v.false_alarm_share:.0%} alerting x {v.cost_per_false_alarm:,.0f}"
            ),
        ),
    ]
    df = pd.DataFrame(rows, columns=["mechanism", "value_per_year", "calculation"])
    total = pd.DataFrame(
        [["Net value per year", df["value_per_year"].sum(), "sum of the rows above"]],
        columns=df.columns,
    )
    return pd.concat([df, total], ignore_index=True)
