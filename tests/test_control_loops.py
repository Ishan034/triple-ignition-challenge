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


def _loop_campaign(seed=0):
    """Hourly pH and base flow for 12 normal batches and one with a failing base pump.

    In normal batches acid bursts pull pH down and the controller answers with more base.
    In the faulty batch the pump stops at 30 h and pH falls with no response."""
    from contamination.schema import MEASUREMENT_COLUMNS

    rng = np.random.default_rng(seed)
    rows, t0 = [], pd.Timestamp("2026-01-01")
    for i in range(13):
        b, start = f"L{i:02d}", t0 + pd.Timedelta(days=3 * i)
        burst = rng.integers(10, 40)
        for h in range(48):
            ph = 6.5 + rng.normal(0, 0.01)
            base = 50 + rng.normal(0, 3)
            if i < 12 and burst <= h < burst + 3:  # disturbance, corrected
                ph -= 0.15
                base += 40
            if i == 12 and 30 <= h < 34:  # pump failure, no correction
                ph -= 0.1 * (h - 29)
                base = 5.0
            ts = start + pd.Timedelta(hours=h)
            rows += [(b, ts, ts, "ph", ph, "pH"), (b, ts, ts, "base_flow", base, "L/h")]
    m = pd.DataFrame(
        rows, columns=["batch_id", "timestamp", "available_at", "variable", "value", "unit"]
    )
    m["stage"], m["source"], m["quality_flag"], m["observation"] = None, "test", "valid", None
    return m[MEASUREMENT_COLUMNS]


def test_loop_check_flags_an_unanswered_drift_only():
    from contamination.ingest import validate_measurements
    from contamination.monitor import fit_reference, monitor

    m = validate_measurements(_loop_campaign()).measurements
    normal = [f"L{i:02d}" for i in range(12)]
    loops = {"ph": ("base_flow", True)}
    ref = fit_reference(m, normal, control_loops=loops)
    assert ref.params["control_loops"] == {"ph": ["base_flow", True]}
    result = monitor(m[m["batch_id"] == "L12"], ref)
    loop_alerts = [a for a in result.alerts if a.detector == "loop_check"]
    assert loop_alerts and min(a.start_h for a in loop_alerts) <= 32
    assert all(a.variable == "ph" and a.direction == "low" for a in loop_alerts)
    # Without the loops the same profile has no loop check at all.
    plain = fit_reference(m, normal)
    assert not [
        a for a in monitor(m[m["batch_id"] == "L12"], plain).alerts if a.detector == "loop_check"
    ]


def _linked_campaign(seed=1):
    """Two linked signals (DO and pH proxies) for 15 normal batches and one where both
    change together from 40 h. Normal batches have their own offset and one-signal wobble."""
    from contamination.schema import MEASUREMENT_COLUMNS

    rng = np.random.default_rng(seed)
    rows, t0 = [], pd.Timestamp("2026-02-01")
    for i in range(16):
        b, start = f"K{i:02d}", t0 + pd.Timedelta(days=3 * i)
        offset_do, offset_ph = rng.normal(0, 5), rng.normal(0, 0.05)
        wobble = rng.integers(20, 60)
        for h in range(80):
            do = 60 + offset_do + rng.normal(0, 1)
            ph = 6.8 + offset_ph + rng.normal(0, 0.01)
            if i < 15 and wobble <= h < wobble + 10:  # one signal wobbles alone
                do -= 4
            if i == 15 and h >= 40:  # both change together, and keep changing
                do -= 0.4 * (h - 39)
                ph -= 0.004 * (h - 39)
            ts = start + pd.Timedelta(hours=h)
            rows += [
                (b, ts, ts, "dissolved_oxygen", do, "%sat"),
                (b, ts, ts, "ph", ph, "pH"),
            ]
    m = pd.DataFrame(
        rows, columns=["batch_id", "timestamp", "available_at", "variable", "value", "unit"]
    )
    m["stage"], m["source"], m["quality_flag"], m["observation"] = None, "test", "valid", None
    return m[MEASUREMENT_COLUMNS]


def test_linked_shift_needs_two_signals_together():
    from contamination.ingest import validate_measurements
    from contamination.monitor import fit_reference, monitor

    m = validate_measurements(_linked_campaign()).measurements
    normal = [f"K{i:02d}" for i in range(15)]
    ref = fit_reference(m, normal, linked_signals=["dissolved_oxygen", "ph"])
    assert set(ref.params["linked_shift"]["scales"]) == {"dissolved_oxygen", "ph"}
    alerts = monitor(m[m["batch_id"] == "K15"], ref).alerts
    linked = [a for a in alerts if a.detector == "linked_shift"]
    assert linked and min(a.start_h for a in linked) >= 40
    assert "dissolved_oxygen" in linked[0].detail and "ph" in linked[0].detail
