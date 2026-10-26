"""Importer: plant-style exports -> campaign tables, and the readiness report."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from contamination.importer import (
    convert,
    convert_mapping,
    draft_mapping,
    readiness,
    register_variables,
    suggest_category,
    suggest_variable,
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


def test_excel_exports(tmp_path):
    pd.DataFrame({"Batch": ["X1"], "Start": ["2026-03-01 08:00"]}).to_excel(
        tmp_path / "b.xlsx", index=False
    )
    pd.DataFrame(
        {"Batch": ["X1", "X1"], "Time": ["2026-03-01 08:00", "2026-03-01 09:00"], "pH": [7.0, 6.9]}
    ).to_excel(tmp_path / "m.xlsx", index=False)
    mapping = {
        "measurements": [
            {"file": "m.xlsx", "batch_id": "Batch", "timestamp": "Time", "columns": {"pH": "ph"}}
        ],
        "batches": {"file": "b.xlsx", "columns": {"batch_id": "Batch", "start_time": "Start"}},
    }
    (tmp_path / "mapping.yaml").write_text(yaml.safe_dump(mapping))
    res = convert(tmp_path / "mapping.yaml", tmp_path / "out")
    assert res.tables["measurements"]["value"].tolist() == [7.0, 6.9]


def test_draft_mapping_recovers_the_template_mapping(exported, tmp_path):
    frames = {f.name: pd.read_csv(f) for f in exported.glob("*.csv")}
    draft = draft_mapping(frames)
    hand = yaml.safe_load((exported / "mapping.yaml").read_text())
    for key in ("batches", "roles"):
        assert draft[key] == hand[key]
    assert draft["outcomes"]["category_map"] == hand["outcomes"]["category_map"]
    by_file = {s["file"]: s for s in draft["measurements"]}
    for spec in hand["measurements"]:
        got = by_file[spec["file"]]
        assert got.get("columns") == spec.get("columns")
        assert got.get("variable_map") == spec.get("variable_map")
    # The draft converts to the same campaign as the hand-written mapping.
    a = convert_mapping(draft, exported, tmp_path / "draft").tables["measurements"]
    b = convert(exported / "mapping.yaml", tmp_path / "hand").tables["measurements"]
    assert len(a) == len(b) and a["value"].sum() == pytest.approx(b["value"].sum())


def test_suggestions_are_conservative():
    assert suggest_variable("pH") == "ph"
    assert suggest_variable("pH offline") == "ph_offline"
    assert suggest_variable("Phase") is None  # 'ph' only as a whole word
    assert suggest_variable("Batch") is None
    assert suggest_variable("dO2", {"dissolved_oxygen": ["dO2"]}) == "dissolved_oxygen"
    assert suggest_category("Contaminated") == "contamination"
    assert suggest_category("OK") == "normal"
    assert suggest_category("???") is None  # left for the user to choose


def test_maverickx_profile_marks_unconfirmed_numbers():
    from contamination.diagnosis import load_knowledge
    from contamination.plant import load_plant

    kb = load_knowledge(profile="maverickx")
    assert kb.roles["ph"] == "ph" and "MaverickX" in kb.review_status
    plant = load_plant("maverickx")
    assert plant.value_defaults["test_cost"].value == 10000
    assert not any(v.confirmed for v in plant.value_defaults.values())
    assert load_plant("indpensim") is None  # no plant section
