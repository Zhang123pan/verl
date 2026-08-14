"""Protocol smoke using an authoritative city movement-route artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .message_router import parse_sender_message, render_receiver_message_context, route_messages
from .prompt_builder import build_receiver_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="jinan")
    parser.add_argument("--route-table", default=None)
    parser.add_argument("--sender", default=None)
    args = parser.parse_args()

    route_path = Path(args.route_table) if args.route_table else (
        Path(__file__).parent / "artifacts" / f"movement_routes_{args.city}.json"
    )
    with route_path.open(encoding="utf-8") as handle:
        route_table = json.load(handle)
    signal_pairs = (("ETWT", ("ET", "WT")), ("NTST", ("NT", "ST")),
                    ("ELWL", ("EL", "WL")), ("NLSL", ("NL", "SL")))
    sender = args.sender
    selected_signal, expected = signal_pairs[0]
    if sender is None:
        for candidate, value in route_table["routes"].items():
            available = value.get("movements", {})
            for signal, pair in signal_pairs:
                if all(not available[m].get("is_boundary") and available[m].get("receiver_id")
                       for m in pair if m in available) and all(m in available for m in pair):
                    sender, selected_signal, expected = candidate, signal, pair
                    break
            if sender:
                break
    else:
        available = route_table["routes"][sender]["movements"]
        for signal, pair in signal_pairs:
            if all(m in available and not available[m].get("is_boundary") and available[m].get("receiver_id")
                   for m in pair):
                selected_signal, expected = signal, pair
                break
    if sender is None:
        raise RuntimeError("No sender has a fully routable signal movement pair")
    movements = [(movement, route_table["routes"][sender]["movements"][movement]) for movement in expected]
    if not movements:
        raise RuntimeError(f"No routable movements for sender {sender}")
    message_lines = "\n".join(
        f'  <to movement="{movement}">{movement} arrivals are approaching.</to>'
        for movement, _ in movements
    )
    response = f"""<perception>{{"candidate_phases": []}}</perception>
<mode>slow</mode>
<reasoning>Smoke protocol message.</reasoning>
<current_v>{{"ETWT": 3, "NTST": 1, "ELWL": 0, "NLSL": 0}}</current_v>
<signal>{selected_signal}</signal>
<message>
{message_lines}
</message>"""
    parsed = parse_sender_message(response, selected_signal=selected_signal)
    routed = route_messages(sender, parsed, route_table)
    contexts = render_receiver_message_context(routed)
    base = [{"role": "system", "content": "receiver smoke system"},
            {"role": "user", "content": "receiver smoke user"}]
    receiver_prompts = {receiver: build_receiver_prompt(base, context)
                        for receiver, context in contexts.items()}
    print(json.dumps({
        "city": args.city,
        "route_table": str(route_path),
        "sender": sender,
        "movements": [movement for movement, _ in movements],
        "parsed": parsed,
        "routed": [item.__dict__ for item in routed],
        "receivers": sorted(contexts),
        "receiver_prompt_count": {key: len(value) for key, value in receiver_prompts.items()},
        "status": "PASS",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
