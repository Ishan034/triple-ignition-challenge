"""Controller-output evidence (E17/E18) and the equipment-fault explanation (H06)."""

from dataclasses import replace

import numpy as np
import pandas as pd

from contamination.alerts import Alert
from contamination.diagnosis import EvidenceItem, _control_loop_evidence, assess, load_knowledge


def _kb():
    kb = load_knowledge()
    return replace(
        kb, roles={**kb.roles, "ph_control": "base_flow", "temperature_control": "not_measured"}
    )


def _scores(ph_z, base_z) -> pd.DataFrame:
    hours = np.arange(len(ph_z), dtype=float)
    rows = [("B1", "ph", h, z) for h, z in zip(hours, ph_z)]
    rows += [("B1", "base_flow", h, z) for h, z in zip(hours, base_z)]
    return pd.DataFrame(rows, columns=["batch_id", "variable", "elapsed_h", "z"])


def _alert(variable, direction, start, end) -> Alert:
    return Alert(
        "B1", variable, "robust_threshold", "", start, end, start, int(end - start + 1),
        "persistent", direction, -9.0, 4.0, 0.0, 0.0, 1.0, -9.0, "t", "r",
    )  # fmt: skip


def test_drift_without_controller_response_points_at_the_actuator():
    kb = _kb()
    s = _scores([0] * 5 + [-8] * 5, [0] * 5 + [-2] * 5)  # pH falls, base flow falls too
    e17, e18 = _control_loop_evidence(s, [_alert("ph", "low", 5, 9)], kb.parameters, kb)
    assert e18[0] == "present" and e17[0] == "absent"


def test_drift_with_controller_pushing_back_is_not_an_actuator_fault():
    kb = _kb()
    s = _scores([0] * 5 + [-8] * 5, [0] * 5 + [6] * 5)  # base pump works hard
    _, e18 = _control_loop_evidence(s, [_alert("ph", "low", 5, 9)], kb.parameters, kb)
    assert e18[0] == "absent"


def test_controller_working_hard_while_holding_its_variable():
    kb = _kb()
    s = _scores([0] * 10, [0] * 5 + [7] * 5)  # pH held, base demand up (e.g. acid production)
    e17, e18 = _control_loop_evidence(s, [_alert("base_flow", "high", 5, 9)], kb.parameters, kb)
    assert e17[0] == "present" and e18[0] == "absent"


def test_unknown_without_controller_outputs():
    kb = load_knowledge()  # the synthetic proxy records no controller outputs
    e17, e18 = _control_loop_evidence(_scores([0] * 4, [0] * 4), [], kb.parameters, kb)
    assert e17[0] == e18[0] == "unknown"


def test_unresponsive_controller_makes_equipment_fault_lead():
    kb = _kb()
    evidence = {e: EvidenceItem(e, kb.evidence[e]["name"], "absent", "") for e in kb.evidence}
    evidence["E18"].state = "present"
    evidence["E14"].state = "present"  # temperature off: a condition change also fits
    hyps = assess(evidence, kb)
    assert hyps[0].id == "H06" and hyps[0].tier == "leading"
    assert next(h for h in hyps if h.id == "H01").score < 0
