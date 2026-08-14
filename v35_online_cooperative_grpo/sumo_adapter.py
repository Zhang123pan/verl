"""Adapters from phase-3 Ray actors to the repository's existing ``SUMOEnv``.

They use SUMOEnv's tested saveState/loadState round trip.  The adapter returns
raw one-hop endpoint statistics; the cooperative reward equation is intentionally
kept outside this module until its design is finalised.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .ray_actors import SnapshotRef
from .observation import V30Observation


def _phase_vehicle_count(intersection: Any, phase: str) -> int:
    """Match V30's 150m Current V accounting for a controlled phase."""
    movements = [phase[i : i + 2] for i in range(0, len(phase), 2)]
    movement_index = {"WL": 0, "WT": 1, "WR": 2, "EL": 3, "ET": 4, "ER": 5,
                      "NL": 6, "NT": 7, "NR": 8, "SL": 9, "ST": 10, "SR": 11}
    sets = intersection.dic_feature.get("traffic_movement_vehicle_ids_150m", [])
    return len({vehicle_id for movement in movements for vehicle_id in (
        sets[movement_index[movement]] if movement in movement_index and movement_index[movement] < len(sets) else []
    )})


def _phase_cycle_count(phase: str, values: list[Any]) -> int:
    movements = [phase[i : i + 2] for i in range(0, len(phase), 2)]
    movement_index = {"WL": 0, "WT": 1, "WR": 2, "EL": 3, "ET": 4, "ER": 5,
                      "NL": 6, "NT": 7, "NR": 8, "SL": 9, "ST": 10, "SR": 11}
    return sum(int(values[movement_index[movement]]) for movement in movements
               if movement in movement_index and movement_index[movement] < len(values))


@dataclass
class V25State:
    histories: dict[str, dict[str, list[int]]] = field(default_factory=dict)


class SUMOEnvAdapter:
    """Persistent actor-local SUMOEnv with phase-label action APIs."""

    def __init__(self, env: Any, seed: int | None = None):
        self.env = env
        self.seed = seed
        self.v25 = V25State()

    def reset(self, seed: int | None = None) -> None:
        self.seed = self.seed if seed is None else int(seed)
        self.env.reset(use_gui=False, seed=self.seed, verbose=False)
        self.v25 = V25State()

    def close(self) -> None:
        self.env.close()

    def save_snapshot(self, path: str) -> float:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.env.snapshot(path) is None:
            raise RuntimeError(f"SUMOEnv failed to save state: {path}")
        return float(self.env.get_current_time())

    def load_snapshot(self, path: str) -> None:
        self.env.load_from_file(path, quiet=True, raise_on_error=True)

    def prepare_snapshot(self, snapshot: SnapshotRef) -> None:
        if snapshot.seed is not None and self.seed is not None and snapshot.seed != self.seed:
            raise RuntimeError("SUMOEnvAdapter cannot restore a snapshot from another seed.")

    def _action_indices(self, actions: Mapping[str, str]) -> dict[str, int]:
        by_id = {inter.inter_id: inter for inter in self.env.list_intersection}
        unknown = set(actions) - set(by_id)
        if unknown:
            raise KeyError(f"Action table has unknown intersections: {sorted(unknown)}")
        result = {}
        for inter_id, inter in by_id.items():
            phase = actions.get(inter_id)
            if phase is None:
                raise KeyError(f"Action table omits intersection {inter_id!r}")
            if phase not in inter.control_phases:
                raise ValueError(f"{inter_id} cannot execute phase {phase!r}")
            result[inter_id] = inter.control_phases.index(phase)
        return result

    def execute(self, actions: dict[str, str], seconds: int) -> dict[str, Any]:
        # SUMOEnv follows the gym-style four-value API: state, reward, done, info.
        step_result = self.env.step(self._action_indices(actions), min_action_time=seconds)
        # Production SUMOEnv returns four values; keep compatibility with the
        # lightweight three-value test/dummy environments used by the suite.
        if len(step_result) == 4:
            state, reward, done, info = step_result
        elif len(step_result) == 3:
            state, done, info = step_result
            reward = 0.0
        else:
            raise ValueError(f"SUMOEnv.step returned {len(step_result)} values")
        return {
            "state": state,
            "reward": reward,
            "done": bool(done),
            "info": info,
            "sim_time_s": float(self.env.get_current_time()),
            "endpoint": self.endpoint_statistics(),
        }

    def endpoint_statistics(self, intersection_ids: tuple[str, ...] | None = None) -> dict[str, dict[str, float]]:
        allowed = set(intersection_ids) if intersection_ids else None
        result = {}
        for inter in self.env.list_intersection:
            if allowed is not None and inter.inter_id not in allowed:
                continue
            vehicle_sets = inter.dic_feature.get("traffic_movement_vehicle_ids_150m", [])
            ids = {vehicle_id for group in vehicle_sets for vehicle_id in group}
            speeds = inter.dic_vehicle_speed_current_step
            result[inter.inter_id] = {
                "remaining_v_150m": float(len(ids)),
                "queue_150m": float(sum(speeds.get(vehicle_id, 1.0) < 0.1 for vehicle_id in ids)),
            }
        return result

    def background_actions(self) -> dict[str, str]:
        """Repository V25 rule: Current V > V35-V > nonzero history > order."""
        actions = {}
        for inter in self.env.list_intersection:
            histories = self.v25.histories.setdefault(inter.inter_id, {phase: [] for phase in inter.control_phases})
            cycle = list(inter.dic_feature.get("v9_cycle_150m_history") or [])
            scored = []
            for order_idx, phase in enumerate(inter.control_phases):
                current_v = _phase_vehicle_count(inter, phase)
                histories.setdefault(phase, []).append(current_v)
                sequence = [_phase_cycle_count(phase, values) for values in cycle]
                trend = sequence[-1] - sequence[0] if len(sequence) >= 2 else 0
                scored.append((current_v, trend, sum(value > 0 for value in histories[phase]), -order_idx, phase))
            selected = max(scored)[-1]
            actions[inter.inter_id] = selected
            histories[selected] = []
        return actions

    def advance_background(self, seconds: int) -> None:
        self.execute(self.background_actions(), seconds)

    def observations(self) -> dict[str, Any] | None:
        """Bare SUMO smoke adapters do not own the V30 renderer/recorder."""
        return None


class V30RecordingMasterAdapter(SUMOEnvAdapter):
    """Master-only adapter that reuses OneLine's V30 renderer callbacks."""

    def __init__(self, runtime: Any, seed: int | None = None):
        super().__init__(runtime.env, seed)
        self.runtime = runtime
        self.decision_step = 0
        self._observations: dict[str, V30Observation] = {}

    def reset(self, seed: int | None = None) -> None:
        self.seed = self.seed if seed is None else int(seed)
        self.env.reset(use_gui=False, seed=self.seed, verbose=False)
        self.runtime._reset_perception_renderer()
        self.v25 = V25State()
        self.decision_step = 0
        self._observations = {}

    def advance_background(self, seconds: int) -> None:
        start_s = float(self.env.get_current_time())
        actions = self.background_actions()
        self.runtime._begin_video_interval(self.decision_step, start_s)
        try:
            self.env.step(
                self._action_indices(actions),
                min_action_time=seconds,
                inner_step_callback=self.runtime._build_perception_snapshot_callback(self.decision_step),
            )
            details = self.runtime._finalize_video_interval(float(self.env.get_current_time()))
        except BaseException as exc:
            self.runtime._finalize_video_interval(
                float(self.env.get_current_time()), status="incomplete",
                error=f"{type(exc).__name__}: {exc}", force_discard=True,
            )
            raise
        end_s = float(self.env.get_current_time())
        paths = details.get("paths", {}) if isinstance(details, dict) else {}
        frame_counts = details.get("frame_counts", {}) if isinstance(details, dict) else {}
        observations = {}
        for inter in self.env.list_intersection:
            videos = paths.get(inter.inter_id, {})
            if set(videos) != {"E", "W", "N", "S"}:
                raise RuntimeError(f"V30 recorder did not produce four videos for {inter.inter_id}: {videos}")
            if int(frame_counts.get(inter.inter_id, 0)) != 6:
                raise RuntimeError(
                    f"V30 recorder expected six frames for {inter.inter_id}, got {frame_counts.get(inter.inter_id)}"
                )
            observation = V30Observation(
                inter.inter_id, self.decision_step, start_s, end_s,
                {direction: videos[direction] for direction in ("E", "W", "N", "S")},
            )
            observation.validate(seconds)
            observations[inter.inter_id] = observation
        self._observations = observations
        self.decision_step += 1

    def observations(self) -> dict[str, V30Observation]:
        return dict(self._observations)

    def close(self) -> None:
        close_capture = getattr(self.runtime, "_close_perception_image_capture", None)
        if callable(close_capture):
            close_capture()
        self.env.close()


class SUMOEnvFactory:
    """Pickle-friendly factory injected into Ray actors by the online launcher.

    ``config_by_city`` and ``paths_by_city`` should be derived from the same
    city settings used by ``run_v18.py``. Each invocation receives a private
    work directory, preventing TraCI port/files from crossing Ray actors.
    """

    def __init__(self, config_by_city: Mapping[str, Mapping[str, Any]], paths_by_city: Mapping[str, Mapping[str, Any]], work_root: str, repo_root: str | None = None):
        self.config_by_city = {key: deepcopy(value) for key, value in config_by_city.items()}
        self.paths_by_city = {key: deepcopy(value) for key, value in paths_by_city.items()}
        self.work_root = work_root
        self.repo_root = str(Path(repo_root).resolve()) if repo_root else None

    def __call__(self, city: str, seed: int, actor_id: str) -> SUMOEnvAdapter:
        if city not in self.config_by_city or city not in self.paths_by_city:
            raise KeyError(f"No SUMO configuration registered for city {city!r}")
        # Ray workers do not inherit the launcher cwd/PYTHONPATH reliably.
        # Inject the repository root before importing the existing SUMOEnv.
        import sys
        if self.repo_root and self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)
        from utils.sumo_env import SUMOEnv

        work_dir = Path(self.work_root) / city / actor_id
        work_dir.mkdir(parents=True, exist_ok=True)
        config, paths = deepcopy(self.config_by_city[city]), deepcopy(self.paths_by_city[city])
        config.update({"USE_GUI": False, "SEED": int(seed), "ENABLE_VIDEO_SFT_EXTRACTION": False,
                       "ENABLE_COUNTERFACTUAL_DISCHARGE_LOG": False, "RAISE_INNER_STEP_CALLBACK_ERRORS": True})
        paths["PATH_TO_WORK_DIRECTORY"] = str(work_dir)
        env = SUMOEnv(str(work_dir), str(work_dir), config, paths, config.get("INTER_PHASE_MAPPING", {}))
        adapter = SUMOEnvAdapter(env, seed)
        adapter.reset(seed)
        return adapter


class V30RecordingMasterFactory(SUMOEnvFactory):
    """Create a Ray-local V30 mainline runtime with renderer and recorder."""

    def __call__(self, city: str, seed: int, actor_id: str) -> V30RecordingMasterAdapter:
        if city not in self.config_by_city or city not in self.paths_by_city:
            raise KeyError(f"No SUMO configuration registered for city {city!r}")
        import sys
        if self.repo_root and self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)
        from utils.oneline import OneLine
        from utils.vlm_config import get_vlm_config

        class RecordingRuntime(OneLine):
            # The online master uses the adapter's V25 background policy; it
            # does not need one Python controller object per intersection.
            def _create_agents(self) -> None:
                self.agents = []

        work_dir = Path(self.work_root) / city / actor_id
        model_dir = work_dir / "models"
        work_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        config = deepcopy(self.config_by_city[city])
        paths = deepcopy(self.paths_by_city[city])
        rendering_backend = os.environ.get("V35_RENDERING_BACKEND", "p3headlessgl")
        if rendering_backend not in {"p3headlessgl", "p3tinydisplay", "pandagl"}:
            raise ValueError(f"Unsupported V35_RENDERING_BACKEND: {rendering_backend!r}")
        vlm_config = get_vlm_config(city)
        print({
            "v35_renderer_city": city,
            "backend": rendering_backend,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ray_preserve_cuda_visible_devices": os.environ.get(
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
            ),
            "scenario_glb_dir": vlm_config.get("SCENARIO_GLB_DIR"),
        })
        vlm_config.update({
            "DECISION_INPUT_MODE": "video",
            "VIDEO_INCLUDE_CURRENT_IMAGES": False,
            "VIDEO_EXPORT_DIRECTIONS": True,
            "VIDEO_EXPORT_COMPOSITE": False,
            "VIDEO_EXPORT_DIRECTION_SEQUENCE": False,
            "VIDEO_RECORD_MODE": "sampled",
            "VIDEO_FRAME_SAMPLE_INTERVAL": 5.0,
            "VIDEO_FPS": 1,
            "VIDEO_SAVE_COORDINATION_FRAMES": True,
            "VIDEO_ASYNC_WRITE": False,
            "RENDER_PRESET": "1080P",
            "RENDERING_BACKEND": rendering_backend,
            "RENDER_KEEP_BATCH_SENSORS": False,
            "RENDER_REUSE_BATCH_SENSORS": True,
            "RENDER_STEP_TASK_MANAGER": False,
            "LOCAL_RENDER_RADIUS_M": 180.0,
        })
        if city == "newyork":
            vlm_config["TLS_BATCH_SIZE"] = 20
        config.update({
            "MODEL_NAME": "V25",
            "USE_GUI": False,
            "SEED": int(seed),
            "ENABLE_VIDEO_SFT_EXTRACTION": True,
            "ENABLE_COUNTERFACTUAL_DISCHARGE_LOG": False,
            "RAISE_INNER_STEP_CALLBACK_ERRORS": True,
            "VLM_CONFIG": vlm_config,
        })
        topology = Path(paths["PATH_TO_DATA"]) / "network_topology.json"
        if topology.is_file():
            config["NETWORK_TOPOLOGY_PATH"] = str(topology)
        paths.update({"PATH_TO_WORK_DIRECTORY": str(work_dir), "PATH_TO_MODEL": str(model_dir)})
        runtime = RecordingRuntime({}, config, paths, city, config["TRAFFIC_FILE"])
        return V30RecordingMasterAdapter(runtime, seed)


class MultiCityBranchAdapter:
    """Rebuild an actor-local SUMOEnv when a mixed-city batch changes identity."""

    def __init__(self, factory: SUMOEnvFactory, actor_id: str):
        self.factory = factory
        self.actor_id = actor_id
        self.adapter: SUMOEnvAdapter | None = None
        self.identity: tuple[str, int | None] | None = None

    def prepare_snapshot(self, snapshot: SnapshotRef) -> None:
        identity = (snapshot.city, snapshot.seed)
        if self.adapter is not None and self.identity == identity:
            return
        if self.adapter is not None:
            self.adapter.close()
        if snapshot.seed is None:
            raise ValueError("Branch snapshot must include the master seed.")
        self.adapter = self.factory(snapshot.city, snapshot.seed, f"branch_{self.actor_id}")
        self.identity = identity

    def load_snapshot(self, path: str) -> None:
        assert self.adapter is not None
        self.adapter.load_snapshot(path)

    def execute(self, actions: dict[str, str], seconds: int) -> dict[str, Any]:
        assert self.adapter is not None
        return self.adapter.execute(actions, seconds)

    def compute_reward(self, region: dict[str, Any], result: dict[str, Any]) -> float:
        return 0.0

    def close(self) -> None:
        if self.adapter is not None:
            self.adapter.close()
