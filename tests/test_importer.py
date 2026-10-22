"""Importer: plant-style exports -> campaign tables, and the readiness report."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from contamination.importer import (
    convert,
    readiness,
    register_variables,
    write_example_exports,
)
from contamination.pipeline import fit, load_campaign, run_campaign
from contamination.schema import VARIABLES

SYNTHETIC = Path("data/synthetic")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("exports")
    return write_example_exports(SYNTHETIC, out)


def test_round_trip_keeps_every_value(exported, tmp_path):
    res = convert(exported / "mapping.yaml", tmp_path / "campaign")
    assert res.unmapped == []
    orig = load_campaign(SYNTHETIC).measurements
    back = load_campaign(tmp_path / "campaign").measurements
    key = ["batch_id", "variable", "timestamp"]
    a = orig[orig["observation"].isna()].drop_duplicates(key).set_index(key)["value"]
    b = back[back["observation"].isna()].drop_duplicates(key).set_index(key)["value"]
    joined = pd.concat([a, b], axis=1, keys=["orig", "back"]).dropna(how="all")
    both = joined.dropna()
    assert len(both) >= 0.99 * a.notna().sum()
    assert np.allclose(both["orig"], both["back"])
    # Missing stays missing: never filled.
    assert not back.loc[back["value"].isna(), "quality_flag"].eq("valid").any()
    # Lab results keep their reporting delay.
    act = back[back["variable"] == "enzyme_activity"]
    assert ((act["available_h"] - act["elapsed_h"]) > 0).all()
    # Operator notes survive as observations.
    assert back["observation"].notna().sum() == orig["observation"].notna().sum()


def test_imported_campaign_runs_end_to_end(exported, tmp_path):
    convert(exported / "mapping.yaml", tmp_path / "campaign")
    camp = load_campaign(tmp_path / "campaign")
    result = run_campaign(camp, fit(camp))
    assert len(result.overview(camp)) == len(camp.monitored_ids)


def test_readiness_reports_gaps(exported, tmp_path):
    convert(exported / "mapping.yaml", tmp_path / "campaign")
    r = readiness(tmp_path / "campaign").set_index("check")
    assert r.loc["biomass", "status"] == "ok"
    assert r.loc["ph_control", "status"] == "warn"  # controller outputs not provided
    assert r.loc["Confirmed contamination cases", "status"] == "warn"  # 6 < 10
    assert "43" in r.loc["Approved normal batches", "detail"]


def test_long_layout_reports_unmapped_names_and_elapsed_hours(tmp_path):
    pd.DataFrame({"Batch": ["X1"], "Start": ["2026-03-01 08:00"], "Vessel": ["V1"]}).to_csv(
        tmp_path / "b.csv", index=False
    )
    pd.DataFrame(
        {
            "Batch": ["X1"] * 4,
            "Hour": [0, 1, 2, 3],
            "Tag": ["PH", "PH", "PH", "FOAM"],
            "Val": [7.0, 6.9, None, 1],
        }
    ).to_csv(tmp_path / "m.csv", index=False)
    mapping = {
        "measurements": [
            {
                "file": "m.csv",
                "layout": "long",
                "batch_id": "Batch",
                "elapsed_hours": "Hour",
                "variable": "Tag",
                "value": "Val",
                "variable_map": {"PH": "ph"},
            }
        ],
        "batches": {
            "file": "b.csv",
            "columns": {"batch_id": "Batch", "start_time": "Start", "bioreactor_id": "Vessel"},
        },
    }
    (tmp_path / "mapping.yaml").write_text(yaml.safe_dump(mapping))
    res = convert(tmp_path / "mapping.yaml", tmp_path / "out")
    m = res.tables["measurements"]
    assert len(m) == 3 and m["value"].isna().sum() == 1
    assert m["timestamp"].iloc[1] == pd.Timestamp("2026-03-01 09:00")
    assert any("FOAM" in u for u in res.unmapped)
    assert (res.tables["batches"]["medium_lot"] == "not recorded").all()


def test_custom_variables_are_registered():
    register_variables(
        {"base_pump_test": {"unit": "%", "min_scale": 0.5, "plausible_range": [0, 100]}}
    )
    assert VARIABLES["base_pump_test"].unit == "%"
    with pytest.raises(ValueError):
        register_variables({"ph": {"unit": "x", "min_scale": 1, "plausible_range": [0, 1]}})
