#!/usr/bin/env python3
"""Verify malformed model outputs receive gate rewards instead of exceptions."""

from v35_offline_grpo_reward import _format, compute_score


CASES = (
    "",
    "<perception>null</perception><mode>fast</mode><current_v>null</current_v><signal>ETWT</signal>",
    "<perception>{}</perception><mode>fast</mode><current_v>{}</current_v><signal>ETWT</signal>",
    "<perception>{\"ETWT\":null,\"NTST\":{},\"ELWL\":{},\"NLSL\":{}}</perception>"
    "<mode>fast</mode><current_v>{\"ETWT\":0,\"NTST\":0,\"ELWL\":0,\"NLSL\":0}</current_v>"
    "<signal>ETWT</signal>",
    "<perception>{\"ETWT\":{\"current_v\":7},\"NTST\":{},\"ELWL\":{},\"NLSL\":{}}</perception>"
    "<mode>fast</mode><current_v>{\"ETWT\":0,\"NTST\":0,\"ELWL\":0,\"NLSL\":0}</current_v>"
    "<signal>ETWT</signal>",
    "<perception>{\"ETWT\":{\"current_v\":3}}</perception><mode>fast</mode>"
    "<current_v>{\"ETWT\":0,\"NTST\":0,\"ELWL\":0,\"NLSL\":0}</current_v><signal>ETWT</signal>",
    "<perception>{bad json}</perception><mode>slow</mode><reasoning>x</reasoning>"
    "<current_v>{}</current_v><signal>NTST</signal>",
)


def main() -> None:
    for index, solution in enumerate(CASES):
        parsed, signal_bad, _ = _format(solution)
        result = compute_score(solution, {})
        score = result["score"]
        assert set(result) == {
            "score", "format_reward", "traffic_reward", "perception_reward", "reasoning_penalty"
        }
        assert not parsed, (index, solution)
        if signal_bad:
            assert score == -1.0, (index, signal_bad, score)
        elif "<mode>slow</mode>" in solution:
            assert -0.3 <= score <= 0.0, (index, signal_bad, score)
        else:
            assert score == 0.0, (index, signal_bad, score)
    print(f"PASS: {len(CASES)} malformed outputs returned gate rewards without exceptions.")


if __name__ == "__main__":
    main()
