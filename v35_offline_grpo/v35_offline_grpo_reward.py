"""Offline V35 GRPO reward: contract gate and counterfactual traffic."""

from __future__ import annotations

import json
import math
import os
import re
import datetime
from functools import lru_cache
from typing import Any

PHASES = ("ETWT", "NTST", "ELWL", "NLSL")
MOVEMENTS = {"ETWT": ("ET", "WT"), "NTST": ("NT", "ST"), "ELWL": ("EL", "WL"), "NLSL": ("NL", "SL")}
# Slow reasoning is free up to the response budget.  Otherwise fast would
# strictly dominate a concise slow response with the same control outcome.
SLOW_FIXED_COST = 0.0
REASONING_FREE_TOKENS = 300
REASONING_TAU = 400.0
REASONING_MAX_LENGTH_PENALTY = 0.10
FAST_DECISION_PENALTY = 0.05


def _write_reward_log(record: dict[str, Any]) -> None:
    """Append one complete reward audit record; safe for concurrent workers."""
    path = os.environ.get("V35_GRPO_REWARD_LOG", "")
    if not path:
        return
    record = dict(record)
    record["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record["pid"] = os.getpid()
    try:
        payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except OSError:
        # Reward logging must never interrupt training.
        pass


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


def _has_exact_tag_pair(text: str, name: str) -> bool:
    return (
        len(re.findall(rf"<{name}>", text or "", flags=re.I)) == 1
        and len(re.findall(rf"</{name}>", text or "", flags=re.I)) == 1
    )


def _phase_map(perception: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(perception, dict):
        return {}
    if all(isinstance(perception.get(p), dict) for p in PHASES):
        return {p: perception[p] for p in PHASES}
    rows = perception.get("candidate_phases")
    if isinstance(rows, list):
        return {r.get("signal"): r for r in rows if isinstance(r, dict) and r.get("signal") in PHASES}
    return {}


def _has_exact_candidate_phases(perception: Any) -> bool:
    if not isinstance(perception, dict):
        return False
    rows = perception.get("candidate_phases")
    if not isinstance(rows, list) or len(rows) != len(PHASES):
        return False
    signals = [row.get("signal") for row in rows if isinstance(row, dict)]
    return len(signals) == len(PHASES) and set(signals) == set(PHASES)


def _valid_phase_row(row: Any, phase: str) -> bool:
    """Validate one perception row without raising on malformed model output."""
    if not isinstance(row, dict):
        return False

    current_v = row.get("current_v")
    current_q = row.get("current_q")
    coordinated = row.get("coordinated_arrivals")
    if not all(isinstance(group, dict) for group in (current_v, current_q, coordinated)):
        return False
    if not all(isinstance(group.get("total"), (int, float)) for group in (current_v, current_q, coordinated)):
        return False
    if not isinstance(row.get("nonzero_v_history_length_since_last_service"), (int, float)):
        return False
    if not isinstance(row.get("demand_trend_v30_minus_v5"), (int, float)):
        return False
    if not isinstance(row.get("queue_trend_q30_minus_q5"), (int, float)):
        return False

    movements = MOVEMENTS[phase]
    if not all(isinstance(current_v.get(movement), (int, float)) for movement in movements):
        return False
    if not all(isinstance(current_q.get(movement), (int, float)) for movement in movements):
        return False
    if current_v["total"] != sum(current_v[movement] for movement in movements):
        return False
    if current_q["total"] != sum(current_q[movement] for movement in movements):
        return False
    if not 0 <= current_q["total"] <= current_v["total"]:
        return False
    return coordinated["total"] >= 0


def _format(solution: str) -> tuple[bool, bool, dict[str, Any]]:
    names = ("perception", "mode", "reasoning", "signal")
    blocks = {n: _tag(solution, n) for n in names}
    required = ("perception", "mode", "signal")
    signal_bad = (
        len(blocks["signal"]) != 1
        or not _has_exact_tag_pair(solution, "signal")
        or blocks["signal"][0].strip() not in PHASES
    )
    if any(len(blocks[name]) != 1 or not _has_exact_tag_pair(solution, name) for name in required):
        return False, signal_bad, {}
    mode = blocks["mode"][0].strip().lower()
    expected_order = ("perception", "mode", "signal") if mode == "fast" else (
        "perception", "mode", "reasoning", "signal"
    )
    positions = [solution.lower().find(f"<{name}>") for name in expected_order]
    order = all(position >= 0 for position in positions) and positions == sorted(positions)
    reasoning_contract = (
        mode == "fast" and not blocks["reasoning"] and not re.search(r"</?reasoning>", solution, flags=re.I)
    ) or (
        mode == "slow"
        and len(blocks["reasoning"]) == 1
        and _has_exact_tag_pair(solution, "reasoning")
        and bool(blocks["reasoning"][0].strip())
    )
    # The full match rejects old <current_v> blocks, code fences, and any
    # additional top-level text/tags instead of silently accepting them.
    if mode == "fast":
        envelope = r"\s*<perception>.*?</perception>\s*<mode>fast</mode>\s*<signal>.*?</signal>\s*"
    else:
        envelope = r"\s*<perception>.*?</perception>\s*<mode>slow</mode>\s*<reasoning>.*?</reasoning>\s*<signal>.*?</signal>\s*"
    exact_envelope = re.fullmatch(envelope, solution or "", flags=re.I | re.S) is not None
    if not order or not reasoning_contract or not exact_envelope:
        return False, signal_bad, {}
    perception = _json(blocks["perception"][0])
    phases = _phase_map(perception)
    valid = mode in ("fast", "slow") and _has_exact_candidate_phases(perception) and len(phases) == 4
    if valid:
        valid = all(_valid_phase_row(phases.get(phase), phase) for phase in PHASES)
    return bool(valid), signal_bad, {"mode": mode, "signal": blocks["signal"][0].strip(), "perception": perception, "reasoning": blocks["reasoning"][0] if blocks["reasoning"] else ""}


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
    candidate = actions.get(signal, {})
    if all(isinstance(candidate.get(k), (int, float)) for k in ("queue_score", "volume_score", "discharge_score")):
        return (
            float(candidate["queue_score"])
            + float(candidate["volume_score"])
            + float(candidate["discharge_score"])
        ) / 3.0
    values = {k: {m: float(actions[k].get(m, 0.0)) for m in ("remaining_v_30s", "remaining_queue_30s", "discharged_30s")} for k in PHASES}
    v = _rank_scores({p: x["remaining_v_30s"] for p, x in values.items()}, False)
    q = _rank_scores({p: x["remaining_queue_30s"] for p, x in values.items()}, False)
    d = _rank_scores({p: x["discharged_30s"] for p, x in values.items()}, True)
    return (v[signal] + q[signal] + d[signal]) / 3.0


def _perception_values(value: Any) -> list[float]:
    """Flatten the 11 scored values for each of the four phase rows (44 total).

    ``is_boundary`` is intentionally excluded.  The two movement entries in
    each current_v/current_q group and the two movement counts in
    coordinated_arrivals are the two breakdown values for that phase.
    """
    phases = _phase_map(value)
    if len(phases) != 4:
        return []
    result: list[float] = []
    for phase in PHASES:
        row = phases.get(phase)
        if not isinstance(row, dict):
            return []
        try:
            movements = MOVEMENTS[phase]
            current_v = row["current_v"]
            current_q = row["current_q"]
            arrivals = row["coordinated_arrivals"]
            result.extend([float(current_v["total"]), *(float(current_v[m]) for m in movements)])
            result.extend([float(current_q["total"]), *(float(current_q[m]) for m in movements)])
            result.extend([float(row["demand_trend_v30_minus_v5"]), float(row["queue_trend_q30_minus_q5"])])
            breakdown = arrivals["breakdown"]
            result.extend([float(arrivals["total"]), *(float(breakdown[m]["count"]) for m in movements)])
        except (KeyError, TypeError, ValueError):
            return []
    return result


def _perception_bad(solution: str, perception_obj: Any) -> bool:
    if len(_perception_values(perception_obj)) != 44:
        return True
    perception_blocks = _tag(solution, "perception")
    return len(perception_blocks) != 1


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    path = os.environ.get("V35_GRPO_MODEL_PATH", "/home/apulis-dev/userdata/VLMTSCS/training/LlamaFactory/saves/qwen35-4b/merged/v35_four_video_context_reasoning")
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _reasoning_penalty(mode: str, reasoning: str) -> float:
    """Penalize only slow reasoning that exceeds the 300-token budget."""
    if mode != "slow":
        return 0.0
    try:
        token_count = len(_tokenizer()(reasoning, add_special_tokens=False)["input_ids"])
    except Exception:
        token_count = len(reasoning.split())
    excess = max(0, token_count - REASONING_FREE_TOKENS)
    length_penalty = REASONING_MAX_LENGTH_PENALTY * (
        1.0 - math.exp(-excess / REASONING_TAU)
    )
    return SLOW_FIXED_COST + length_penalty


def compute_score(solution_str, ground_truth, **kwargs):
    gt = _json(ground_truth) or {}
    solution = solution_str or ""
    extra_info = kwargs.get("extra_info") or {}
    split = extra_info.get("split", "unknown") if isinstance(extra_info, dict) else "unknown"
    sample_id = extra_info.get("index") if isinstance(extra_info, dict) else None
    parsed, signal_bad, output = _format(solution)
    mode_blocks = _tag(solution, "mode")
    reasoning_blocks = _tag(solution, "reasoning")
    mode = mode_blocks[0].strip().lower() if len(mode_blocks) == 1 else ""
    reasoning = reasoning_blocks[0] if len(reasoning_blocks) == 1 else ""
    reasoning_penalty = _reasoning_penalty(mode, reasoning)

    if signal_bad:
        result = {
            "score": -1.0,
            "format_reward": 0.0,
            "traffic_reward": 0.0,
            "perception_reward": 0.0,
            "reasoning_penalty": reasoning_penalty,
        }
        _write_reward_log({"split": split, "sample_id": sample_id, "solution": solution,
                           "ground_truth": gt, "parsed": parsed, "mode": output.get("mode"),
                           "signal": output.get("signal"), "signal_bad": True,
                           "perception_bad": False, **result})
        return result

    if not parsed:
        perception_blocks = _tag(solution, "perception")
        signal_blocks = _tag(solution, "signal")
        output = {
            "mode": mode,
            "signal": signal_blocks[0].strip(),
            "perception": _json(perception_blocks[0]) if len(perception_blocks) == 1 else {},
            "reasoning": reasoning,
        }

    perception_bad = _perception_bad(solution, output.get("perception"))
    if perception_bad:
        result = {
            "score": -1.0,
            "format_reward": 0.0,
            "traffic_reward": 0.0,
            "perception_reward": 0.0,
            "reasoning_penalty": reasoning_penalty,
        }
        _write_reward_log({"split": split, "sample_id": sample_id, "solution": solution,
                           "ground_truth": gt, "parsed": parsed, "mode": output.get("mode"),
                           "signal": output.get("signal"), "signal_bad": False,
                           "perception_bad": True, **result})
        return result

    format_reward = 1.0 if parsed else 0.0
    predicted_values = _perception_values(output.get("perception"))
    target_values = _perception_values(gt.get("perception_target"))
    correct = sum(abs(pred - target) <= 1 for pred, target in zip(predicted_values, target_values))
    perception_reward = correct / 44.0 if len(predicted_values) == 44 and len(target_values) == 44 else 0.0
    traffic_reward = _traffic(gt, output["signal"])
    actions = gt.get("actions", {})
    action_rewards = {
        phase: _traffic(gt, phase) for phase in PHASES
    } if isinstance(actions, dict) and set(actions) == set(PHASES) else {}
    best_traffic_reward = max(action_rewards.values(), default=traffic_reward)
    # Fast is intended for an unambiguous quick decision.  Penalize only a
    # strictly suboptimal fast action; ties for the best counterfactual remain
    # valid. Slow already receives the natural traffic reward and is not
    # double-penalized here.
    fast_decision_penalty = (
        FAST_DECISION_PENALTY
        if output.get("mode") == "fast" and traffic_reward < best_traffic_reward
        else 0.0
    )
    base_reward = traffic_reward + 0.5 * perception_reward - reasoning_penalty - fast_decision_penalty
    format_term = -0.5 if not parsed else 0.0
    score = base_reward + format_term
    result = {
        "score": score,
        "format_reward": format_reward,
        "format_term": format_term,
        "base_reward": base_reward,
        "traffic_reward": traffic_reward,
        "perception_reward": perception_reward,
        "reasoning_penalty": reasoning_penalty,
        "best_traffic_reward": best_traffic_reward,
        "fast_decision_penalty": fast_decision_penalty,
        "perception_correct": correct,
        "perception_total": 44,
    }
    _write_reward_log({"split": split, "sample_id": sample_id, "solution": solution,
                       "ground_truth": gt, "parsed": parsed, "mode": output.get("mode"),
                       "signal": output.get("signal"), "signal_bad": signal_bad,
                       "perception_bad": perception_bad, **result})
    return result
