"""Build V35 perception-only SFT targets and segmented rollout masks."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any


PHASES = ("ETWT", "NTST", "ELWL", "NLSL")
MOVEMENTS = {
    "ETWT": ("ET", "WT"),
    "NTST": ("NT", "ST"),
    "ELWL": ("EL", "WL"),
    "NLSL": ("NL", "SL"),
}
ALLOWED_LANES = {
    "ETWT": "East straight and West straight lanes",
    "NTST": "North straight and South straight lanes",
    "ELWL": "East left and West left lanes",
    "NLSL": "North left and South left lanes",
}
CURRENT_PHASE_PATTERN = re.compile(r"\bCurrent phase:\s*(ETWT|NTST|ELWL|NLSL)\b", re.I)


def _as_python(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _message_text(messages: Any) -> str:
    messages = _as_python(messages)
    if not isinstance(messages, (list, tuple)):
        return ""
    chunks: list[str] = []
    for message in messages:
        message = _as_python(message)
        if not isinstance(message, dict):
            continue
        content = _as_python(message.get("content"))
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, (list, tuple)):
            for part in content:
                part = _as_python(part)
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
    return "\n".join(chunks)


def build_perception_sft_row_indices(uids: Any, rollout_n: int) -> tuple[list[int], list[int]]:
    """Return gold source rows and group-preserving rollout/gold row order."""
    if rollout_n <= 0:
        raise ValueError(f"rollout_n must be positive, got {rollout_n}")

    groups: dict[str, list[int]] = {}
    for index, raw_uid in enumerate(uids):
        uid = str(_as_python(raw_uid))
        groups.setdefault(uid, []).append(index)
    if not groups:
        raise ValueError("perception SFT requires at least one uid group")

    bad_groups = {uid: len(indices) for uid, indices in groups.items() if len(indices) != rollout_n}
    if bad_groups:
        preview = dict(list(bad_groups.items())[:5])
        raise ValueError(f"perception SFT expected {rollout_n} rollouts per uid, got {preview}")

    rollout_count = sum(len(indices) for indices in groups.values())
    gold_indices: list[int] = []
    interleaved_indices: list[int] = []
    for gold_offset, indices in enumerate(groups.values()):
        gold_indices.append(indices[0])
        interleaved_indices.extend(indices)
        interleaved_indices.append(rollout_count + gold_offset)
    return gold_indices, interleaved_indices


def _ground_truth(reward_model: Any) -> dict[str, Any]:
    reward_model = _as_python(reward_model)
    if not isinstance(reward_model, dict):
        raise ValueError("perception SFT requires reward_model")
    value = _as_python(reward_model.get("ground_truth"))
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("perception SFT requires reward_model.ground_truth")
    return value


def build_perception_target(reward_model: Any, raw_prompt: Any) -> str:
    """Return the canonical perception-only assistant continuation."""
    target = _ground_truth(reward_model).get("perception_target")
    if not isinstance(target, dict) or set(target) != set(PHASES):
        raise ValueError("perception SFT target must contain exactly four phases")

    match = CURRENT_PHASE_PATTERN.search(_message_text(raw_prompt))
    if match is None:
        raise ValueError("could not find 'Current phase' in the V35 prompt")

    candidate_phases = []
    for phase in PHASES:
        row = deepcopy(target[phase])
        if not isinstance(row, dict):
            raise ValueError(f"invalid perception target row for {phase}")
        candidate_phases.append(
            {
                "signal": phase,
                "allowed_lanes": ALLOWED_LANES[phase],
                "current_v": row["current_v"],
                "current_q": row["current_q"],
                "demand_trend_v30_minus_v5": row["demand_trend_v30_minus_v5"],
                "queue_trend_q30_minus_q5": row["queue_trend_q30_minus_q5"],
                "coordinated_arrivals": row["coordinated_arrivals"],
                "nonzero_v_history_length_since_last_service": row[
                    "nonzero_v_history_length_since_last_service"
                ],
            }
        )

    payload = {
        "current_phase": match.group(1).upper(),
        "candidate_phases": candidate_phases,
    }
    return "<perception>\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n</perception>"


def find_subsequence(values: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(values):
        return None
    limit = len(values) - len(needle) + 1
    for index in range(limit):
        if values[index : index + len(needle)] == needle:
            return index
    return None


def build_grpo_loss_mask(response_ids: list[int], tokenizer) -> list[int]:
    """Mask policy gradients to the decision suffix beginning at ``<mode>``.

    Malformed outputs without a mode tag retain a full mask so the format
    penalty can suppress the sampled sequence. Gold perception CE supplies the
    positive correction for the perception prefix.
    """
    mode_tokens = tokenizer.encode("<mode>", add_special_tokens=False)
    start = find_subsequence(response_ids, mode_tokens)
    if start is None:
        return [1] * len(response_ids)
    return [0] * start + [1] * (len(response_ids) - start)
