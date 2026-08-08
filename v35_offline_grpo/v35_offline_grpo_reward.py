"""Offline V35 GRPO reward: contract gate, counterfactual traffic, and Current V."""

from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from typing import Any

PHASES = ("ETWT", "NTST", "ELWL", "NLSL")
MOVEMENTS = {"ETWT": ("ET", "WT"), "NTST": ("NT", "ST"), "ELWL": ("EL", "WL"), "NLSL": ("NL", "SL")}
TAU = 300.0
P_MAX = 0.3


def _json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _tag(text: str, name: str) -> list[str]:
    return re.findall(rf"<{name}>\s*(.*?)\s*</{name}>", text or "", flags=re.I | re.S)


def _phase_map(perception: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(perception, dict):
        return {}
    if all(isinstance(perception.get(p), dict) for p in PHASES):
        return {p: perception[p] for p in PHASES}
    rows = perception.get("candidate_phases")
    if isinstance(rows, list):
        return {r.get("signal"): r for r in rows if isinstance(r, dict) and r.get("signal") in PHASES}
    return {}


def _format(solution: str) -> tuple[bool, bool, dict[str, Any]]:
    names = ("perception", "mode", "reasoning", "current_v", "signal")
    blocks = {n: _tag(solution, n) for n in names}
    signal_bad = len(blocks["signal"]) != 1 or blocks["signal"][0].strip() not in PHASES
    positions = [solution.lower().find(f"<{n}>") for n in ("perception", "mode", "current_v", "signal")]
    order = all(x >= 0 for x in positions) and positions == sorted(positions)
    if not order or any(len(blocks[n]) != 1 for n in ("perception", "mode", "current_v", "signal")):
        return False, signal_bad, {}
    mode = blocks["mode"][0].strip().lower()
    perception = _json(blocks["perception"][0])
    current_v = _json(blocks["current_v"][0])
    phases = _phase_map(perception)
    valid = mode in ("fast", "slow") and isinstance(current_v, dict) and set(current_v) == set(PHASES) and len(phases) == 4
    for phase in PHASES:
        row = phases.get(phase, {})
        for group in ("current_v", "current_q", "coordinated_arrivals"):
            valid &= isinstance(row.get(group), dict) and isinstance(row[group].get("total"), (int, float))
        valid &= isinstance(row.get("nonzero_v_history_length_since_last_service"), (int, float))
        valid &= isinstance(row.get("demand_trend_v30_minus_v5"), (int, float))
        valid &= isinstance(row.get("queue_trend_q30_minus_q5"), (int, float))
        valid &= all(isinstance(row.get("current_v", {}).get(m), (int, float)) for m in MOVEMENTS[phase])
        valid &= all(isinstance(row.get("current_q", {}).get(m), (int, float)) for m in MOVEMENTS[phase])
        valid &= row.get("current_v", {}).get("total") == sum(row.get("current_v", {}).get(m, 0) for m in MOVEMENTS[phase])
        valid &= row.get("current_q", {}).get("total") == sum(row.get("current_q", {}).get(m, 0) for m in MOVEMENTS[phase])
        valid &= 0 <= row.get("current_q", {}).get("total", -1) <= row.get("current_v", {}).get("total", -1)
        valid &= row.get("coordinated_arrivals", {}).get("total", -1) >= 0
    valid &= all(isinstance(current_v[p], int) and current_v[p] >= 0 for p in PHASES)
    valid &= all(current_v[p] == phases[p].get("current_v", {}).get("total") for p in PHASES)
    valid &= (mode == "fast" and len(blocks["reasoning"]) == 0) or (mode == "slow" and len(blocks["reasoning"]) == 1 and bool(blocks["reasoning"][0].strip()))
    return bool(valid), signal_bad, {"mode": mode, "signal": blocks["signal"][0].strip(), "perception": perception, "current_v": current_v, "reasoning": blocks["reasoning"][0] if blocks["reasoning"] else ""}


def _rank_scores(values: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    if len(set(values.values())) == 1:
        return {p: 0.5 for p in PHASES}
    order = sorted(PHASES, key=lambda p: values[p], reverse=higher_is_better)
    base = (1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0)
    scores: dict[str, float] = {}
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        scores_for_tie = base[i:j]
        for p in order[i:j]:
            scores[p] = sum(scores_for_tie) / len(scores_for_tie)
        i = j
    return scores


def _traffic(ground_truth: dict[str, Any], signal: str) -> float:
    actions = ground_truth.get("actions", {})
    if set(actions) != set(PHASES):
        return 0.0
    values = {k: {m: float(actions[k].get(m, 0.0)) for m in ("remaining_v_30s", "remaining_queue_30s", "discharged_30s")} for k in PHASES}
    v = _rank_scores({p: x["remaining_v_30s"] for p, x in values.items()}, False)
    q = _rank_scores({p: x["remaining_queue_30s"] for p, x in values.items()}, False)
    d = _rank_scores({p: x["discharged_30s"] for p, x in values.items()}, True)
    return (v[signal] + q[signal] + d[signal]) / 3.0


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    path = os.environ.get("V35_GRPO_MODEL_PATH", "/home/apulis-dev/userdata/VLMTSCS/training/LlamaFactory/saves/qwen35-4b/merged/v35_four_video_context_reasoning")
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def compute_score(solution_str, ground_truth, **kwargs):
    gt = _json(ground_truth) or {}
    parsed, signal_bad, output = _format(solution_str or "")
    if signal_bad:
        return -1.0
    if not parsed:
        return 0.0
    target = gt.get("perception_target", {})
    correct = 0
    for phase in PHASES:
        predicted = output["perception"]["candidate_phases"] if isinstance(output["perception"], dict) and isinstance(output["perception"].get("candidate_phases"), list) else output["perception"]
        row = next((r for r in predicted if r.get("signal") == phase), None) if isinstance(predicted, list) else predicted.get(phase)
        value = row.get("current_v", {}).get("total") if isinstance(row, dict) else None
        final = output["current_v"].get(phase)
        if isinstance(value, (int, float)) and isinstance(final, (int, float)) and abs(value - target.get(phase, 10**9)) <= 1 and abs(final - target.get(phase, 10**9)) <= 1:
            correct += 1
    perception_reward = correct / 4.0
    traffic_reward = _traffic(gt, output["signal"])
    try:
        length = len(_tokenizer()(output["reasoning"], add_special_tokens=False)["input_ids"])
    except Exception:
        length = len(output["reasoning"].split())
    reasoning_penalty = 0.0 if output["mode"] == "fast" else P_MAX * (1.0 - math.exp(-length / TAU))
    return traffic_reward + 0.5 * perception_reward - reasoning_penalty

