import pytest

from v35_online_cooperative_grpo.config import OnlineGRPOConfig


def test_config_derives_capacity_without_fixing_it():
    cfg = OnlineGRPOConfig(group_batch_size=4, rollout_n=6)
    assert cfg.requested_branch_actors == 24
    assert cfg.active_branch_actors == 24


def test_config_allows_ray_throttling():
    cfg = OnlineGRPOConfig(group_batch_size=4, rollout_n=6, branch_actor_limit=8)
    assert cfg.requested_branch_actors == 24
    assert cfg.active_branch_actors == 8


def test_first_version_keeps_one_period_reward():
    with pytest.raises(ValueError):
        OnlineGRPOConfig(control_period_s=30, reward_horizon_s=60)
