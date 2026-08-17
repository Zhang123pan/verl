import json

from v35_online_cooperative_grpo.decision_parser import resolve_decision


def _perception():
    return {"candidate_phases": [
        {"signal": phase, "current_v": {"total": index}}
        for index, phase in enumerate(("ETWT", "NTST", "ELWL", "NLSL"))
    ]}


def test_invalid_signal_uses_v25_signal():
    decision = resolve_decision("<perception>{}</perception>", "NLSL")
    assert decision.signal == "NLSL"
    assert decision.signal_source == "v25_fallback"


def test_invalid_perception_keeps_valid_model_signal():
    decision = resolve_decision("<perception>{}</perception><signal>NTST</signal>", "ETWT")
    assert decision.signal == "NTST"
    assert decision.signal_source == "model_signal"
    assert not decision.perception_valid


def test_valid_response_uses_model_signal_and_validates_perception():
    response = f"<perception>{json.dumps(_perception())}</perception><signal>ELWL</signal>"
    decision = resolve_decision(response, "ETWT")
    assert decision.signal == "ELWL"
    assert decision.signal_source == "model_signal"
    assert decision.perception_valid
