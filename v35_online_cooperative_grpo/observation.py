"""V30-compatible media observation contract for online cooperative GRPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VIDEO_DIRECTIONS = ("E", "W", "N", "S")


@dataclass(frozen=True)
class V30Observation:
    """One focal-intersection observation captured before branching.

    The four videos are the just-completed 30-second master window. Upstream
    frames are selected by the existing V30 coordination algorithm and keyed
    by the receiver entry direction.
    """

    intersection_id: str
    decision_step: int
    sim_start_s: float
    sim_end_s: float
    videos: dict[str, str]
    coordination_frames: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    base_messages: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def validate(self, control_period_s: int = 30) -> None:
        if set(self.videos) != set(VIDEO_DIRECTIONS):
            raise ValueError(f"Expected E/W/N/S videos, got {sorted(self.videos)}")
        if abs((self.sim_end_s - self.sim_start_s) - control_period_s) > 1e-6:
            raise ValueError("Observation window must match the V30 control period.")
        missing = [path for path in self.videos.values() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing observation videos: {missing}")
        for frame in self.coordination_frames:
            direction = frame.get("target_entry_direction") or frame.get("direction")
            if direction not in VIDEO_DIRECTIONS:
                raise ValueError(f"Invalid coordination target direction: {direction!r}")
            path = frame.get("path") or frame.get("coordination_frame_path")
            if not path or not Path(path).is_file():
                raise FileNotFoundError(f"Missing coordination frame: {path!r}")
        if self.base_messages and not any(message.get("role") == "user" for message in self.base_messages):
            raise ValueError("Observation base_messages must contain a user message.")
