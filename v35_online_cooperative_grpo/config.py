"""Configuration helpers for Ray-managed online cooperative GRPO."""

from __future__ import annotations

from dataclasses import dataclass

from .ray_actors import branch_actor_count


@dataclass(frozen=True)
class OnlineGRPOConfig:
    group_batch_size: int = 4
    rollout_n: int = 6
    branch_actor_limit: int | None = None
    control_period_s: int = 30
    reward_horizon_s: int = 30
    master_stride_s: int = 30

    def __post_init__(self) -> None:
        if self.reward_horizon_s != self.control_period_s:
            raise ValueError("First version requires reward_horizon_s == control_period_s.")
        if self.master_stride_s != self.control_period_s:
            raise ValueError("Master timeline must advance by one control period.")
        branch_actor_count(self.group_batch_size, self.rollout_n, self.branch_actor_limit)

    @property
    def requested_branch_actors(self) -> int:
        return self.group_batch_size * self.rollout_n

    @property
    def active_branch_actors(self) -> int:
        return branch_actor_count(self.group_batch_size, self.rollout_n, self.branch_actor_limit)

