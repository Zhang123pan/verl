import pytest

from v35_online_cooperative_grpo.trajectory_to_verl import convert_rows, expand_branch


def _prompt(name):
    return [{"role": "user", "content": name}]


def _videos(name):
    return {direction: f"{name}_{direction}.mp4" for direction in ("E", "W", "N", "S")}


def _branch(with_policy=False):
    metadata = None
    if with_policy:
        metadata = {
            "prompt_ids": [1, 2], "response_ids": [3, 4],
            "response_logprobs": [-0.1, -0.2], "response_mask": [1, 1],
        }
    return {
        "snapshot_id": "s1", "branch_id": 2, "focal_id": "sender",
        "reward": -3.0, "advantage": 1.25, "group_size": 6,
        "signal": "NTST", "prompt": _prompt("sender"), "response": "sender response",
        "videos": _videos("sender"), "policy_metadata": metadata,
        "receivers": [{
            "intersection_id": "receiver", "signal": "ETWT",
            "prompt": _prompt("receiver"), "response": "receiver response",
            "videos": _videos("receiver"), "policy_metadata": metadata,
        }],
    }


def test_branch_expands_to_independent_sender_and_receiver_spans():
    records = expand_branch(_branch())
    assert [record["role"] for record in records] == ["sender", "receiver"]
    assert {record["advantage"] for record in records} == {1.25}
    assert {record["branch_uid"] for record in records} == {"s1:branch:2"}
    assert len({record["span_uid"] for record in records}) == 2
    assert all(record["uid"] == "s1" for record in records)
    assert not any(record["policy_ready"] for record in records)


def test_policy_metadata_is_preserved_per_span():
    records = convert_rows([_branch(with_policy=True)], require_policy_metadata=True)
    assert all(record["policy_ready"] for record in records)
    assert all(record["response_ids"] == [3, 4] for record in records)
    assert all(record["response_logprobs"] == [-0.1, -0.2] for record in records)


def test_external_api_text_cannot_be_mislabeled_as_on_policy():
    with pytest.raises(ValueError, match="lack on-policy"):
        convert_rows([_branch()], require_policy_metadata=True)


def test_partial_policy_metadata_is_rejected():
    branch = _branch()
    branch["policy_metadata"] = {"prompt_ids": [1]}
    with pytest.raises(ValueError, match="partial policy metadata"):
        expand_branch(branch)
