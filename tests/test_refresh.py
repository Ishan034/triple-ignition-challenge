"""Reference refresh workflow and the semi-synthetic contaminant."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from contamination.datasets.contaminate import growth, inject
from contamination.refresh import HISTORY, apply, plan, refreshed_ids

SYNTHETIC = Path("data/synthetic")


def test_refresh_proposes_only_clean_batches_and_records_history(tmp_path):
    d = tmp_path / "campaign"
    shutil.copytree(SYNTHETIC, d)
    (d / "reference_batches.txt").write_text(
        "\n".join(pd.read_csv(d / "batches.csv").sort_values("start_time")["batch_id"][:30])
    )
    p = plan(d)
    outcomes = pd.read_csv(d / "outcomes.csv").set_index("batch_id")
    chosen = p.candidates.loc[p.candidates["candidate"], "batch_id"]
    assert len(chosen) and (outcomes.loc[chosen, "category"] == "normal").all()
    assert set(p.added) == set(chosen) and not p.removed
    assert {"variable", "median_shift_sd", "flag"} <= set(p.drift.columns)
    apply(d, p, "test review")
    history = pd.read_csv(d / HISTORY)
    assert list(history["note"]) == ["initial reference", "test review"]
    assert (d / "reference_batches.txt").read_text().split() == p.new_ids


def test_rolling_window_keeps_the_latest():
    assert refreshed_ids(["a", "b"], ["d"], ["a", "b", "c", "d"], keep_latest=2) == ["b", "d"]


def test_contaminant_effects_start_at_onset():
    t = np.arange(0.0, 60.0)
    g = growth(t, 20.0)
    assert (g[t < 20] == 0).all() and g[20] < 0.05 and g[32] == 0.5 and g[-1] > 0.99
    batch = pd.DataFrame(
        {
            "batch_id": "P1",
            "variable": np.repeat(["cer", "do2_mg_l"], len(t)),
            "elapsed_h": np.tile(t, 2),
            "value": 1.0,
        }
    )
    out = inject(batch, 20.0, 1.0)
    cer, do = out[out["variable"] == "cer"], out[out["variable"] == "do2_mg_l"]
    assert (cer.loc[cer["elapsed_h"] < 20, "value"] == 1.0).all()
    assert cer["value"].iloc[-1] > 1.19 and do["value"].iloc[-1] < 0.76
