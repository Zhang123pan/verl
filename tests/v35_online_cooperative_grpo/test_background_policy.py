from v35_online_cooperative_grpo.background_policy import choose_actions, choose_phase


def row(v, trend=0, history=0):
    return {"current_v": {"total": v}, "demand_trend_v30_minus_v5": trend,
            "nonzero_v_history_length_since_last_service": history}


def test_v25_priority_order():
    assert choose_phase({"ETWT": row(3, 0, 0), "NTST": row(2, 9, 9), "ELWL": row(1), "NLSL": row(0)}) == "ETWT"
    assert choose_phase({"ETWT": row(2, 0), "NTST": row(2, 1), "ELWL": row(2, 1, 4), "NLSL": row(0)}) == "ELWL"


def test_fixed_phase_order_breaks_complete_tie():
    observation = {phase: row(0) for phase in ("ETWT", "NTST", "ELWL", "NLSL")}
    assert choose_phase(observation) == "ETWT"


def test_builds_full_master_action_table():
    assert choose_actions({"a": {"ETWT": row(1)}, "b": {"NTST": row(2)}}) == {"a": "ETWT", "b": "NTST"}
