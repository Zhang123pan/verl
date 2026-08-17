"""Serialization helpers for online cooperative GRPO rollout groups."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def summarize_smoke(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return protocol and reward diagnostics for one cooperative smoke batch."""
    branches = len(records)
    receiver_rows = [receiver for row in records for receiver in row.get("receivers", [])]
    rewards = [float(row["reward"]) for row in records if "reward" in row]
    reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
    reward_std = (
        math.sqrt(sum((reward - reward_mean) ** 2 for reward in rewards) / len(rewards))
        if rewards else 0.0
    )
    return {
        "branches": branches,
        "sender_signal_valid": sum(bool(row.get("sender_signal_valid")) for row in records),
        "sender_message_valid": sum(bool(row.get("sender_message_valid")) for row in records),
        "branches_with_receivers": sum(bool(row.get("receiver_ids")) for row in records),
        "receiver_generations": len(receiver_rows),
        "receiver_signal_valid": sum(not bool(row.get("fallback")) for row in receiver_rows),
        "fallback_branches": sum(bool(row.get("fallback")) for row in records),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "zero_reward_variance": bool(rewards) and reward_std <= 1e-6,
    }


def write_group_batch(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write one JSONL record per branch with within-snapshot advantages."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["snapshot_id"]), []).append(record)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for snapshot_id, group in groups.items():
            rewards = [float(item["reward"]) for item in group]
            mean = sum(rewards) / len(rewards)
            variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
            std = math.sqrt(variance)
            for item in group:
                row = dict(item)
                row["group_size"] = len(group)
                row["group_reward_mean"] = mean
                row["group_reward_std"] = std
                row["advantage"] = (float(item["reward"]) - mean) / max(std, 1e-6)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
