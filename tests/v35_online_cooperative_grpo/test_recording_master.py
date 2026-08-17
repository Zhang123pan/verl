from pathlib import Path

from v35_online_cooperative_grpo.sumo_adapter import V30RecordingMasterAdapter


class Intersection:
    inter_id = "i"
    control_phases = ["ETWT", "NTST", "ELWL", "NLSL"]
    dic_feature = {"traffic_movement_vehicle_ids_150m": [[]] * 12, "v9_cycle_150m_history": []}
    dic_vehicle_speed_current_step = {}


class Env:
    list_intersection = [Intersection()]
    time = 0
    def get_current_time(self): return self.time
    def step(self, actions, min_action_time, inner_step_callback):
        self.time += min_action_time
        return {}, 0.0, False, {}
    def close(self): pass


class Runtime:
    def __init__(self, root): self.env, self.root = Env(), root
    def _begin_video_interval(self, step, start): pass
    def _build_perception_snapshot_callback(self, step): return lambda *args: None
    def _finalize_video_interval(self, end, **kwargs):
        paths = {}
        for direction in ("E", "W", "N", "S"):
            path = self.root / f"{direction}.mp4"; path.write_bytes(b"video"); paths[direction] = str(path)
        return {"paths": {"i": paths}, "frame_counts": {"i": 6}}
    def _close_perception_image_capture(self): pass


def test_recording_master_attaches_six_frame_v30_window(tmp_path):
    adapter = V30RecordingMasterAdapter(Runtime(tmp_path), seed=1)
    adapter.advance_background(30)
    observation = adapter.observations()["i"]
    assert observation.sim_start_s == 0
    assert observation.sim_end_s == 30
    assert set(observation.videos) == {"E", "W", "N", "S"}


def test_planned_v25_actions_are_reused_by_next_advance(tmp_path):
    adapter = V30RecordingMasterAdapter(Runtime(tmp_path), seed=1)
    calls = 0
    original = adapter.background_actions

    def counted_background_actions():
        nonlocal calls
        calls += 1
        return original()

    adapter.background_actions = counted_background_actions
    planned = adapter.planned_background_actions()
    assert adapter.planned_background_actions() == planned
    adapter.advance_background(30)
    assert calls == 1
