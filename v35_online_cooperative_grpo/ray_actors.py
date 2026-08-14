"""Ray-managed SUMO timeline and counterfactual branch actors.

No threads or Python multiprocessing are used here.  A concrete SUMO adapter
is injected into each actor; Ray owns actor placement and process lifetime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class SnapshotRef:
    city: str
    episode_id: str
    snapshot_id: str
    path: str
    sim_time_s: float
    seed: int | None = None
    observations: dict[str, Any] | None = None


@dataclass(frozen=True)
class BranchRequest:
    snapshot: SnapshotRef
    actions: dict[str, str]
    reward_region: dict[str, Any]
    horizon_s: int = 30


class TimelineAdapter(Protocol):
    def advance_background(self, seconds: int) -> None: ...
    def save_snapshot(self, path: str) -> float: ...
    def close(self) -> None: ...
    def observations(self) -> dict[str, Any]: ...


class BranchAdapter(Protocol):
    def prepare_snapshot(self, snapshot: SnapshotRef) -> None: ...
    def load_snapshot(self, path: str) -> None: ...
    def execute(self, actions: dict[str, str], seconds: int) -> dict[str, Any]: ...
    def compute_reward(self, region: dict[str, Any], result: dict[str, Any]) -> float: ...
    def close(self) -> None: ...


def branch_actor_count(group_batch_size: int, rollout_n: int, max_parallel: int | None = None) -> int:
    """Return requested Ray branch capacity, never fewer than one actor."""
    if group_batch_size < 1 or rollout_n < 2:
        raise ValueError("group_batch_size must be >=1 and rollout_n must be >=2")
    requested = group_batch_size * rollout_n
    return max(1, min(requested, max_parallel)) if max_parallel is not None else requested


def _ray():
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - exercised only without Ray
        raise RuntimeError("Ray is required for online cooperative SUMO actors") from exc
    return ray


def create_actor_classes():
    """Create remote classes lazily so importing this package does not require Ray."""
    ray = _ray()

    @ray.remote(num_cpus=1)
    class MasterTimelineActor:
        def __init__(self, adapter_factory: Callable[[], TimelineAdapter], city: str, episode_id: str, snapshot_dir: str):
            self.adapter = adapter_factory()
            self.city = city
            self.episode_id = episode_id
            self.snapshot_dir = snapshot_dir
            self.counter = 0
            os.makedirs(snapshot_dir, exist_ok=True)

        def publish_snapshot(self) -> SnapshotRef:
            snapshot_id = f"{self.episode_id}_t{self.counter:08d}"
            path = os.path.join(self.snapshot_dir, f"{snapshot_id}.xml")
            sim_time = float(self.adapter.save_snapshot(path))
            if not os.path.isfile(path) or os.path.getsize(path) <= 0:
                raise RuntimeError(f"SUMO snapshot was not created: {path}")
            observations = self.adapter.observations() if hasattr(self.adapter, "observations") else None
            return SnapshotRef(self.city, self.episode_id, snapshot_id, path, sim_time, observations=observations)

        def advance_and_publish(self, seconds: int = 30) -> SnapshotRef:
            if seconds <= 0:
                raise ValueError("seconds must be positive")
            self.adapter.advance_background(seconds)
            self.counter += 1
            return self.publish_snapshot()

        def close(self) -> None:
            self.adapter.close()

    @ray.remote(num_cpus=1)
    class RotatingMasterTimelineActor:
        """One Ray-managed city timeline that restarts at its next seed at 3600 s."""
        def __init__(self, adapter_factory, city: str, seeds: list[int], snapshot_dir: str, episode_seconds: int = 3600):
            if not seeds:
                raise ValueError("A rotating master requires at least one seed.")
            self.adapter_factory, self.city, self.seeds = adapter_factory, city, [int(x) for x in seeds]
            self.snapshot_dir, self.episode_seconds = snapshot_dir, int(episode_seconds)
            self.seed_index, self.episode_index, self.elapsed_s, self.counter = -1, -1, 0, 0
            os.makedirs(snapshot_dir, exist_ok=True)
            self._start_next_episode()

        def _start_next_episode(self) -> None:
            if hasattr(self, "adapter"):
                self.adapter.close()
            self.seed_index = (self.seed_index + 1) % len(self.seeds)
            self.episode_index += 1
            self.seed = self.seeds[self.seed_index]
            self.episode_id = f"{self.city}_seed{self.seed}_episode{self.episode_index:06d}"
            self.adapter = self.adapter_factory(self.city, self.seed, self.episode_id)
            self.elapsed_s = 0

        def publish_snapshot(self) -> SnapshotRef:
            snapshot_id = f"{self.episode_id}_t{self.counter:08d}"
            path = os.path.join(self.snapshot_dir, f"{snapshot_id}.xml")
            sim_time = float(self.adapter.save_snapshot(path))
            if not os.path.isfile(path) or os.path.getsize(path) <= 0:
                raise RuntimeError(f"SUMO snapshot was not created: {path}")
            observations = self.adapter.observations() if hasattr(self.adapter, "observations") else None
            return SnapshotRef(self.city, self.episode_id, snapshot_id, path, sim_time, self.seed, observations)

        def advance_and_publish(self, seconds: int = 30) -> SnapshotRef:
            if seconds <= 0 or seconds > self.episode_seconds:
                raise ValueError("Invalid master advance duration.")
            if self.elapsed_s + seconds > self.episode_seconds:
                self._start_next_episode()
            self.adapter.advance_background(seconds)
            self.elapsed_s += seconds
            self.counter += 1
            return self.publish_snapshot()

        def status(self) -> dict[str, Any]:
            return {"city": self.city, "seed": self.seed, "episode_id": self.episode_id, "elapsed_s": self.elapsed_s}

        def close(self) -> None:
            self.adapter.close()

    @ray.remote(num_cpus=1)
    class SumoBranchActor:
        def __init__(self, adapter_factory: Callable[[], BranchAdapter], actor_index: int):
            self.adapter = adapter_factory()
            self.actor_index = actor_index

        def rollout(self, request: BranchRequest, branch_id: int) -> dict[str, Any]:
            if request.horizon_s <= 0:
                raise ValueError("horizon_s must be positive")
            self.adapter.prepare_snapshot(request.snapshot)
            self.adapter.load_snapshot(request.snapshot.path)
            result = self.adapter.execute(request.actions, request.horizon_s)
            reward = float(self.adapter.compute_reward(request.reward_region, result))
            return {
                "branch_id": branch_id,
                "actor_index": self.actor_index,
                "snapshot_id": request.snapshot.snapshot_id,
                "sim_time_s": request.snapshot.sim_time_s,
                "actions": dict(request.actions),
                "reward": reward,
                "environment_result": result,
            }

        def close(self) -> None:
            self.adapter.close()

    return MasterTimelineActor, SumoBranchActor, RotatingMasterTimelineActor


def launch_branch_pool(
    adapter_factory: Callable[[], BranchAdapter],
    group_batch_size: int,
    rollout_n: int,
    max_parallel: int | None = None,
) -> list[Any]:
    """Launch a Ray actor pool; explicit capacity may throttle execution in waves."""
    _MasterTimelineActor, SumoBranchActor, _RotatingMasterTimelineActor = create_actor_classes()
    count = branch_actor_count(group_batch_size, rollout_n, max_parallel)
    return [SumoBranchActor.remote(adapter_factory, i) for i in range(count)]
