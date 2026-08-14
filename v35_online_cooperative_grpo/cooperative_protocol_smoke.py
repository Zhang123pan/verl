"""Protocol-only sender/receiver smoke for online cooperation."""

from __future__ import annotations

import argparse
import json

from .message_router import (
    parse_sender_message,
    render_receiver_message_context,
    route_messages,
)
from .prompt_builder import build_receiver_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender", default="intersection_1_1")
    parser.add_argument("--receiver-a", default="intersection_1_2")
    parser.add_argument("--receiver-b", default="intersection_2_1")
    args = parser.parse_args()

    response = """<perception>{"candidate_phases": []}</perception>
<mode>slow</mode>
<reasoning>Smoke protocol message.</reasoning>
<current_v>{"ETWT": 3, "NTST": 1, "ELWL": 0, "NLSL": 0}</current_v>
<signal>ETWT</signal>
<message>
  <to movement="ET">East straight arrivals are approaching.</to>
  <to movement="WT">West straight arrivals are approaching.</to>
</message>"""
    parsed = parse_sender_message(response, selected_signal="ETWT")
    route_table = {
        "routes": {
            args.sender: {
                "movements": {
                    "ET": {"receiver_id": args.receiver_a, "receiver_entry_direction": "W", "is_boundary": False},
                    "WT": {"receiver_id": args.receiver_b, "receiver_entry_direction": "E", "is_boundary": False},
                }
            }
        }
    }
    routed = route_messages(args.sender, parsed, route_table)
    contexts = render_receiver_message_context(routed)
    base = [
        {"role": "system", "content": "receiver smoke system"},
        {"role": "user", "content": "receiver smoke user"},
    ]
    receiver_prompts = {
        receiver: build_receiver_prompt(base, context)
        for receiver, context in contexts.items()
    }
    print(json.dumps({
        "parsed": parsed,
        "routed": [item.__dict__ for item in routed],
        "receivers": sorted(contexts),
        "receiver_prompt_count": {key: len(value) for key, value in receiver_prompts.items()},
        "status": "PASS",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
