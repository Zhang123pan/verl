from types import SimpleNamespace

import pytest

from verl.trainer.ppo.v1.agent_loop_tq import _is_cooperative_output_group


def _output(advantage=None, reward=1.0):
    extra_fields = {} if advantage is None else {"cooperative_advantage": advantage}
    return SimpleNamespace(extra_fields=extra_fields, reward_score=reward)


def test_cooperative_group_requires_all_spans_and_preserves_individual_rewards():
    outputs = [_output(1.0, -0.2), _output(1.0, -0.8)]
    assert _is_cooperative_output_group(outputs)
    assert [item.reward_score for item in outputs] == [-0.2, -0.8]


def test_ordinary_group_uses_existing_postprocessing():
    assert not _is_cooperative_output_group([_output(), _output()])


def test_group_rejects_mixed_contracts():
    with pytest.raises(ValueError, match="cannot mix cooperative"):
        _is_cooperative_output_group([_output(1.0), _output()])


def test_group_requires_branch_rewards():
    with pytest.raises(ValueError, match="branch reward"):
        _is_cooperative_output_group([_output(1.0), _output(1.0, None)])
