from v35_online_cooperative_grpo.trajectory import summarize_smoke


def test_smoke_summary_reports_protocol_and_reward_variation():
    records = [
        {
            "reward": -2.0,
            "sender_signal_valid": True,
            "sender_message_valid": True,
            "receiver_ids": ("b",),
            "receivers": [{"fallback": False}],
            "fallback": False,
        },
        {
            "reward": -4.0,
            "sender_signal_valid": True,
            "sender_message_valid": False,
            "receiver_ids": (),
            "receivers": [],
            "fallback": True,
        },
    ]
    summary = summarize_smoke(records)
    assert summary["branches"] == 2
    assert summary["sender_message_valid"] == 1
    assert summary["branches_with_receivers"] == 1
    assert summary["receiver_signal_valid"] == 1
    assert summary["fallback_branches"] == 1
    assert summary["reward_mean"] == -3.0
    assert summary["reward_std"] == 1.0
    assert summary["zero_reward_variance"] is False
