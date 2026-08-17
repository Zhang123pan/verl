from v35_online_cooperative_grpo.ray_actors import SnapshotRef, BranchRequest, branch_actor_count


def test_branch_capacity_is_derived_from_batch_and_rollout():
    assert branch_actor_count(4, 6) == 24
    assert branch_actor_count(4, 6, max_parallel=10) == 10
    assert branch_actor_count(1, 6) == 6


def test_branch_request_is_snapshot_scoped():
    ref = SnapshotRef("jinan", "ep1", "ep1_t00000000", "/shared/s.xml", 0.0, 7)
    req = BranchRequest(ref, {"intersection_1_1": "ETWT"}, {"tls": ["intersection_1_1"]})
    assert req.snapshot.snapshot_id == "ep1_t00000000"
    assert req.horizon_s == 30
    assert req.snapshot.seed == 7


def test_snapshot_background_actions_can_be_shared_without_mutation():
    ref = SnapshotRef(
        "jinan", "ep1", "ep1_t00000000", "/shared/s.xml", 0.0, 7,
        background_actions={"a": "ETWT", "b": "NTST"},
    )
    first = dict(ref.background_actions)
    second = dict(ref.background_actions)
    first["a"] = "ELWL"
    assert second == {"a": "ETWT", "b": "NTST"}
