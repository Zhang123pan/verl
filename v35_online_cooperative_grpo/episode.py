"""One-cycle cooperative rollout orchestration.

The module deliberately owns the GRPO grouping semantics, while SUMO and vLLM
remain adapters.  Six branches must start from the same snapshot and use the
same background actions; only sender/receiver model outputs may differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .message_router import (
    MessageProtocolError,
    RoutedMessage,
    parse_sender_message,
    render_receiver_message_context,
    route_messages,
)
from .prompt_builder import build_receiver_prompt, build_sender_prompt


class CooperativeEnvironment(Protocol):
    """Minimal SUMO adapter required for one cloned 30-second branch."""

    def clone(self, snapshot_id: str, branch_id: int) -> "CooperativeEnvironment": ...

    def base_messages(self, intersection_id: str) -> list[dict[str, Any]]: ...

    def background_actions(self, snapshot_id: str) -> dict[str, str]: ...

    def execute_cycle(self, actions: dict[str, str], seconds: int) -> dict[str, Any]: ...

    def local_reward(self, focal_id: str, receiver_ids: tuple[str, ...], result: dict[str, Any]) -> float: ...


class Policy(Protocol):
    """Policy adapter returning raw sampled text and its rollout metadata."""

    def generate(self, messages: list[dict[str, Any]], *, branch_id: int, role: str) -> "GeneratedResponse": ...


@dataclass(frozen=True)
class GeneratedResponse:
    text: str
    rollout_metadata: dict[str, Any]


@dataclass(frozen=True)
class GroupContext:
    city: str
    snapshot_id: str
    focal_sender: str
    sender_group: str
    route_table: dict[str, Any]
    background_actions: dict[str, str]
    rollout_n: int = 6
    control_seconds: int = 30


@dataclass(frozen=True)
class TrajectorySpan:
    role: str
    intersection_id: str
    response: GeneratedResponse


@dataclass(frozen=True)
class BranchResult:
    branch_id: int
    actions: dict[str, str]
    routed_messages: tuple[RoutedMessage, ...]
    trajectories: tuple[TrajectorySpan, ...]
    reward: float
    environment_result: dict[str, Any]


def _signal(text: str) -> str:
    """Extract exactly one signal; sender message parsing validates its domain."""
    import re

    found = re.findall(r"<signal>\s*(.*?)\s*</signal>", text, re.I | re.S)
    if len(found) != 1:
        raise MessageProtocolError("Response must contain exactly one <signal> tag.")
    return found[0].strip()


def _sender_action_and_messages(text: str) -> tuple[str, dict[str, str]]:
    """Parse sender output once without trusting model-provided destinations."""
    signal = _signal(text)
    return signal, parse_sender_message(text, selected_signal=signal)


class CooperativeEpisode:
    """Build a fixed six-branch GRPO group for one network snapshot."""

    def __init__(self, environment: CooperativeEnvironment, policy: Policy, context: GroupContext):
        if context.rollout_n < 2:
            raise ValueError("GRPO requires at least two counterfactual branches.")
        self.environment = environment
        self.policy = policy
        self.context = context

    def run_branch(self, branch_id: int) -> BranchResult:
        env = self.environment.clone(self.context.snapshot_id, branch_id)
        sender_id = self.context.focal_sender
        sender = self.policy.generate(
            build_sender_prompt(env.base_messages(sender_id)), branch_id=branch_id, role="sender"
        )
        sender_signal, parsed_message = _sender_action_and_messages(sender.text)
        routed = tuple(route_messages(sender_id, parsed_message, self.context.route_table))
        receiver_contexts = render_receiver_message_context(list(routed))

        actions = dict(self.context.background_actions)
        actions[sender_id] = sender_signal
        spans = [TrajectorySpan("sender", sender_id, sender)]
        for receiver_id in sorted(receiver_contexts):
            receiver = self.policy.generate(
                build_receiver_prompt(env.base_messages(receiver_id), receiver_contexts[receiver_id]),
                branch_id=branch_id,
                role="receiver",
            )
            receiver_signal = _signal(receiver.text)
            actions[receiver_id] = receiver_signal
            spans.append(TrajectorySpan("receiver", receiver_id, receiver))

        result = env.execute_cycle(actions, self.context.control_seconds)
        receiver_ids = tuple(sorted(receiver_contexts))
        return BranchResult(branch_id, actions, routed, tuple(spans), env.local_reward(sender_id, receiver_ids, result), result)

    def run_group(self) -> list[BranchResult]:
        """Return branches with identical snapshot and background-action context."""
        return [self.run_branch(branch_id) for branch_id in range(self.context.rollout_n)]
