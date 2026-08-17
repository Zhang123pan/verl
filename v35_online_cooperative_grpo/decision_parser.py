"""Strict V35 response parsing with the same fallback boundary as VideoAgent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


PHASES = ("ETWT", "NTST", "ELWL", "NLSL")


@dataclass(frozen=True)
class ParsedDecision:
    signal: str
    signal_source: str
    perception_valid: bool
    perception_error: str | None = None


def parse_signal(text: str) -> str:
    signals = re.findall(r"<signal>\s*(ETWT|NTST|ELWL|NLSL)\s*</signal>", text, re.I)
    if len(signals) != 1:
        raise ValueError("response must contain exactly one signal block")
    return signals[0].upper()


def parse_perception(text: str) -> dict[str, int]:
    blocks = re.findall(r"<perception>\s*(.*?)\s*</perception>", text, re.I | re.S)
    if len(blocks) != 1:
        raise ValueError("response must contain exactly one perception block")
    perception = json.loads(blocks[0])
    candidates = perception.get("candidate_phases")
    if not isinstance(candidates, list) or len(candidates) != len(PHASES):
        raise ValueError("perception.candidate_phases must contain exactly four entries")
    current_v: dict[str, int] = {}
    for expected, candidate in zip(PHASES, candidates, strict=True):
        if not isinstance(candidate, dict) or candidate.get("signal") != expected:
            raise ValueError("perception candidates must be ordered ETWT, NTST, ELWL, NLSL")
        value = candidate.get("current_v")
        total = value.get("total") if isinstance(value, dict) else None
        if isinstance(total, bool) or not isinstance(total, (int, float)) or total < 0:
            raise ValueError(f"perception candidate {expected} has invalid current_v.total")
        current_v[expected] = int(total)
    return current_v


def resolve_decision(text: str, v25_signal: str) -> ParsedDecision:
    """Match deployment semantics: only an invalid signal changes the action."""
    if v25_signal not in PHASES:
        raise ValueError(f"invalid V25 fallback signal: {v25_signal!r}")
    try:
        signal = parse_signal(text)
        signal_source = "model_signal"
    except Exception:
        signal = v25_signal
        signal_source = "v25_fallback"
    try:
        parse_perception(text)
        return ParsedDecision(signal, signal_source, True)
    except Exception as exc:
        return ParsedDecision(signal, signal_source, False, f"{type(exc).__name__}: {exc}")
