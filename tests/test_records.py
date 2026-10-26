"""Saved check results and alert feedback."""

import pandas as pd
import pytest

from contamination import records


def _fb(verdict, note=None):
    return pd.DataFrame(
        [["B1", "ph", "robust_threshold", 12.0, verdict, note]], columns=records.FEEDBACK_COLUMNS
    )


def test_feedback_latest_verdict_wins(tmp_path):
    records.save_feedback(
        "data/x", pd.concat([_fb("not sure"), _fb("false alarm", "probe")]), tmp_path
    )
    fb = records.load_feedback("data/x", tmp_path)
    assert len(fb) == 1 and fb["verdict"].iloc[0] == "false alarm"
    assert (tmp_path / "data_x" / records.FEEDBACK).exists()


def test_unknown_verdict_is_refused(tmp_path):
    with pytest.raises(ValueError):
        records.save_feedback("data/x", _fb("maybe"), tmp_path)


def test_checks_round_trip_and_restore(tmp_path):
    row = pd.DataFrame(
        [["B1", "T01", "positive", 30.0, "rods seen"]], columns=records.INVESTIGATION_COLUMNS
    )
    records.save_checks("data/x", row, tmp_path)
    back = records.load_checks("data/x", tmp_path)
    merged = records.merge_upload(back, pd.concat([row, row]), records.INVESTIGATION_COLUMNS)
    assert len(merged) == 1
    with pytest.raises(ValueError):
        records.merge_upload(back, row.drop(columns=["result"]), records.INVESTIGATION_COLUMNS)


def test_nothing_recorded_yet(tmp_path):
    assert records.load_checks("data/new", tmp_path).empty
    assert list(records.load_feedback("data/new", tmp_path).columns) == records.FEEDBACK_COLUMNS


def test_actions_log_opens_and_closes(tmp_path):
    records.add_actions(
        "data/x",
        [{"batch_id": "B1", "action": "request check", "detail": "T01"},
         {"batch_id": "B2", "action": "flag at risk", "detail": "medium lot M-1"}],
        tmp_path,
    )  # fmt: skip
    records.add_actions(
        "data/x", [{"batch_id": "B1", "action": "escalate", "detail": ""}], tmp_path
    )
    df = records.load_actions("data/x", tmp_path)
    assert df["action_id"].tolist() == [1, 2, 3] and (df["status"] == "open").all()
    records.set_action_status("data/x", 2, "done", tmp_path)
    df = records.load_actions("data/x", tmp_path)
    assert df.set_index("action_id").loc[2, "status"] == "done"
    with pytest.raises(ValueError):
        records.add_actions(
            "data/x", [{"batch_id": "B1", "action": "delete", "detail": ""}], tmp_path
        )
