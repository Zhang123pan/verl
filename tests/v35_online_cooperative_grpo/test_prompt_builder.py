import json
from pathlib import Path

from v35_online_cooperative_grpo.prompt_builder import (
    BASE_LAYOUT,
    SENDER_MESSAGE_RULES,
    SENDER_LAYOUT,
    build_receiver_prompt,
    build_sender_prompt,
)


def test_multimodal_sender_preserves_media_items():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "prefix\n" + BASE_LAYOUT},
        {"type": "video_url", "video_url": {"url": "x"}},
        {"type": "image_url", "image_url": {"url": "y"}},
    ]}]
    result = build_sender_prompt(messages)
    assert result[0]["content"][0]["text"].count("<message>") >= 1
    assert result[0]["content"][1:] == messages[0]["content"][1:]


def test_multimodal_receiver_appends_context_to_text_only():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "prompt"},
        {"type": "video_url", "video_url": {"url": "x"}},
    ]}]
    result = build_receiver_prompt(messages, "routed")
    assert "routed" in result[0]["content"][0]["text"]
    assert result[0]["content"][1] == messages[0]["content"][1]


def base_messages():
    root = Path(__file__).parents[2]
    row = json.loads((root / "../../grpo_v30_offline_local_video_dataset_reduced_pixels/train_2000.jsonl").resolve().read_text().splitlines()[0])
    return row["prompt"]


def user_content(messages):
    return next(message["content"] for message in messages if message["role"] == "user")


def test_sender_preserves_baseline_except_layout_and_adds_message_protocol():
    original = user_content(base_messages())
    result = user_content(build_sender_prompt(base_messages()))
    assert BASE_LAYOUT not in result
    assert SENDER_LAYOUT in result
    assert "Online cooperation sender requirement:" in result
    assert result.replace(SENDER_LAYOUT, BASE_LAYOUT).removesuffix(SENDER_MESSAGE_RULES) == original


def test_receiver_keeps_baseline_and_only_appends_current_cycle_context():
    original = base_messages()
    context = "Current-cycle routed coordination messages:\n- Source intersection: intersection_1_1"
    result = build_receiver_prompt(original, context)
    assert user_content(result).startswith(user_content(original))
    assert context in user_content(result)
    assert "Do not output a <message> tag." in user_content(result)
    assert original == base_messages()
