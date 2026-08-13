"""Deterministic V25-style background policy for the master timeline."""

from __future__ import annotations

from typing import Any, Mapping

PHASE_ORDER = ("ETWT", "NTST", "ELWL", "NLSL")


def _number(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _phase_row(observation: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    """Accept either a phase-keyed map or the V35 candidate_phases list."""
    if isinstance(observation.get(phase), Mapping):
        return observation[phase]
    rows = observation.get("candidate_phases")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping) and row.get("signal") == phase:
                return row
    return {}


def choose_phase(observation: Mapping[str, Any]) -> str:
    """Choose ``Current V > V35-V > history > fixed phase order``.

    ``V35-V`` is the V35 field ``demand_trend_v30_minus_v5``.  The final
    negative order index makes ties deterministic and preserves the declared
    phase order (ETWT first).
    """
    scored = []
    for order_idx, phase in enumerate(PHASE_ORDER):
        row = _phase_row(observation, phase)
        current_v = row.get("current_v", {})
        if isinstance(current_v, Mapping):
            current_total = current_v.get("total", 0)
        else:
            current_total = current_v
        history = row.get("nonzero_v_history_length_since_last_service", 0)
        scored.append((
            _number(current_total),
            _number(row.get("demand_trend_v30_minus_v5")),
            _number(history),
            -order_idx,
            phase,
        ))
    return max(scored)[-1]


def choose_actions(observations: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Build a complete deterministic action table for every intersection."""
    return {intersection_id: choose_phase(observation) for intersection_id, observation in observations.items()}

