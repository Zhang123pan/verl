"""Serialization helpers for online cooperative GRPO rollout groups."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


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
