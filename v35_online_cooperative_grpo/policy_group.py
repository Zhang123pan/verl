"""Finalize local-policy cooperative branches for verl actor training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyBranch:
    branch_id: int
    mode: str
    reward: float
    outputs: list[Any]


def finalize_policy_group(branches: list[PolicyBranch], rollout_n: int = 6) -> list[Any]:
    """Attach branch-normalized advantages without weighting by receiver count."""
    if len(branches) != rollout_n:
        raise ValueError(f"cooperative group requires {rollout_n} branches, got {len(branches)}")
    branch_ids = [int(branch.branch_id) for branch in branches]
    if sorted(branch_ids) != list(range(rollout_n)):
        raise ValueError(f"branch ids must be exactly 0..{rollout_n - 1}, got {branch_ids}")
    modes = [str(branch.mode).lower() for branch in branches]
    if rollout_n == 6 and (modes.count("fast") != 3 or modes.count("slow") != 3):
        raise ValueError(f"six-branch cooperative group requires FAST/SLOW 3:3, got {modes}")
    if any(not branch.outputs for branch in branches):
        raise ValueError("every cooperative branch must contain at least one policy span")

    rewards = [float(branch.reward) for branch in branches]
    if not all(math.isfinite(reward) for reward in rewards):
        raise ValueError(f"cooperative rewards must be finite, got {rewards}")
    mean = sum(rewards) / rollout_n
    std = math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / rollout_n)

    flattened = []
    for branch, reward in zip(branches, rewards, strict=True):
        advantage = (reward - mean) / max(std, 1e-6)
        for span_index, output in enumerate(branch.outputs):
            if output.response_logprobs is None:
                raise ValueError(
                    f"branch {branch.branch_id} span {span_index} lacks local rollout logprobs"
                )
            if len(output.response_ids) != len(output.response_logprobs):
                raise ValueError(
                    f"branch {branch.branch_id} span {span_index} token/logprob lengths differ"
                )
            output.reward_score = reward
            output.extra_fields.update(
                {
                    "cooperative_advantage": advantage,
                    "cooperative_branch_id": int(branch.branch_id),
                    "cooperative_mode": str(branch.mode).lower(),
                    "cooperative_group_reward_mean": mean,
                    "cooperative_group_reward_std": std,
                    "cooperative_span_index": span_index,
                }
            )
            flattened.append(output)
    return flattened
