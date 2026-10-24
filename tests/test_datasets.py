import numpy as np
import pandas as pd

from contamination.datasets import erythromycin, indpensim
from contamination.diagnosis import load_knowledge


def test_indpensim_fault_windows_and_signatures():
    t = np.round(np.arange(0.2, 60.01, 0.2), 1)
    flags = ((t >= 20) & (t <= 24)).astype(int)
    windows = indpensim.fault_windows(t, flags)
    assert windows == [(20.0, 24.0)]
    assert indpensim.FAULT_SIGNATURES[windows[0]] == "aeration rate fault"


def test_indpensim_reference_split_uses_each_control_strategy():
    ref = indpensim.reference_batches()
    assert len(ref) == 60
    strategies = {indpensim.control_strategy(int(b[1:])) for b in ref}
    assert strategies == {"recipe", "operator", "apc"}
    assert not any(int(b[1:]) > 90 for b in ref)  # no fault batch in the reference


def test_profiles_load():
    assert load_knowledge(profile="indpensim").roles["oxygen"] == "do2_mg_l"
    assert load_knowledge(profile="erythromycin").roles["biomass"] == "not_measured"


def test_erythromycin_convert_keeps_elapsed_hours(tmp_path):
    rows = []
    for b, first_hh in ((1, 30), (2, 34)):
        for i in range(3):
            rows.append(
                {
                    "date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=b, hours=i),
                    "batch_id": b,
                    "hh": first_hh + i,
                    "te": 31.0,
                    "phzx": 7.2,
                    "phlx": 7.2,
                    "doz": -1 if i == 1 else 30,
                    "zs": 100,
                    "our": 30.0,
                    "cer": 25.0,
                    "hx": 5000 + i,
                }
            )
    raw = tmp_path / "EFP_long.csv"
    pd.DataFrame(rows).to_csv(raw, index=False)
    out = erythromycin.convert(raw, tmp_path / "campaign")
    batches = out["batches"].set_index("batch_id")
    first_log = out["measurements"].groupby("batch_id")["timestamp"].min()
    assert ((first_log - batches["start_time"]) / pd.Timedelta(hours=1)).to_dict() == {
        "E001": 30.0,
        "E002": 34.0,
    }
    assert out["outcomes"].set_index("batch_id")["final_potency"].to_dict() == {
        "E001": 5002,
        "E002": 5002,
    }

    from contamination.pipeline import load_campaign

    campaign = load_campaign(tmp_path / "campaign", reference_ids=["E001"])
    m = campaign.measurements
    assert m.loc[m["batch_id"] == "E002", "elapsed_h"].min() == 34.0  # counted from batch start
    assert (
        m.loc[(m["variable"] == "dissolved_oxygen") & (m["value"] < 0), "quality_flag"] == "suspect"
    ).all()


def test_campaign_profile_can_follow_a_packaged_profile(tmp_path):
    from contamination.diagnosis import load_knowledge

    (tmp_path / "profile.yaml").write_text("use: indpensim\n")
    kb = load_knowledge(profile=tmp_path / "profile.yaml")
    assert kb.roles["biomass"] == "biomass_offline"
    assert "cer" in kb.linked_signals and "indpensim" in kb.review_status.lower()
