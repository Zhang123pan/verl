from v35_offline_grpo.perception_sft import build_perception_sft_row_indices


def test_row_indices_restore_noncontiguous_rollout_groups():
    uids = [uid for _ in range(6) for uid in ("a", "b", "c", "d")]

    gold_indices, interleaved = build_perception_sft_row_indices(uids, rollout_n=6)

    assert gold_indices == [0, 1, 2, 3]
    assert interleaved == [
        0,
        4,
        8,
        12,
        16,
        20,
        24,
        1,
        5,
        9,
        13,
        17,
        21,
        25,
        2,
        6,
        10,
        14,
        18,
        22,
        26,
        3,
        7,
        11,
        15,
        19,
        23,
        27,
    ]

