from types import SimpleNamespace

import pytest

from v35_online_cooperative_grpo.policy_group import PolicyBranch, finalize_policy_group


def _span():
    return SimpleNamespace(
        response_ids=[1, 2],
        response_logprobs=[-0.1, -0.2],
        reward_score=None,
        extra_fields={},
    )


def _branches(rewards=(0, 1, 2, 3, 4, 5)):
    modes = ("fast", "slow", "fast", "slow", "fast", "slow")
    return [PolicyBranch(i, modes[i], reward, [_span(), _span()]) for i, reward in enumerate(rewards)]


def test_advantage_is_normalized_over_branches_then_shared_by_spans():
    branches = _branches()
    outputs = finalize_policy_group(branches)
    assert len(outputs) == 12
    for branch in branches:
        first, second = branch.outputs
        assert first.reward_score == second.reward_score == branch.reward
        assert first.extra_fields["cooperative_advantage"] == second.extra_fields["cooperative_advantage"]
    advantages = [branch.outputs[0].extra_fields["cooperative_advantage"] for branch in branches]
    assert sum(advantages) == pytest.approx(0.0)
    assert sum(value * value for value in advantages) / 6 == pytest.approx(1.0)


def test_requires_three_fast_and_three_slow():
    branches = _branches()
    branches[0].mode = "slow"
    with pytest.raises(ValueError, match="3:3"):
        finalize_policy_group(branches)


def test_requires_local_rollout_logprobs():
    branches = _branches()
    branches[2].outputs[1].response_logprobs = None
    with pytest.raises(ValueError, match="lacks local rollout logprobs"):
        finalize_policy_group(branches)


def test_receiver_count_does_not_change_branch_normalization():
    baseline = _branches()
    expanded = _branches()
    expanded[5].outputs.extend([_span(), _span(), _span()])
    finalize_policy_group(baseline)
    finalize_policy_group(expanded)
    assert [b.outputs[0].extra_fields["cooperative_advantage"] for b in baseline] == pytest.approx(
        [b.outputs[0].extra_fields["cooperative_advantage"] for b in expanded]
    )
