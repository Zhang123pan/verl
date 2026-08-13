"""Build the three SUMOEnv configurations used by phase-3 Ray actors."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


CITY_SPECS = {
    "jinan": ("Jinan", "3_4", "jinan_phase.net.xml", "jinan.rou.xml", "jinan.sumocfg", "jinan_phase_mapping.json"),
    "hangzhou": ("Hangzhou", "4_4", "hangzhou_phase.net.xml", "hangzhou.rou.xml", "hangzhou.sumocfg", "hangzhou_phase_mapping.json"),
    "newyork": ("NewYork", "28_7", "newyork_phase.net.xml", "newyork.rou.xml", "newyork.sumocfg", "newyork_phase_mapping.json"),
}


def build_city_env_configs(repo_root: str | Path, episode_seconds: int = 3600) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return config/path maps compatible with ``SUMOEnvFactory``.

    The phase mapping and four-phase V25 state features match the V30 mainline
    configuration in ``run_v18.py``. Video extraction is deliberately disabled
    here; a later observation adapter owns V35 media rendering.
    """
    root = Path(repo_root).resolve()
    configs, paths = {}, {}
    for city, (template, roadnet, net, route, sumocfg, mapping_file) in CITY_SPECS.items():
        data_dir = root / "data" / template / roadnet
        mapping = json.loads((data_dir / mapping_file).read_text(encoding="utf-8"))
        rows, cols = (int(value) for value in roadnet.split("_"))
        configs[city] = {
            "NUM_AGENTS": rows * cols,
            "NUM_INTERSECTIONS": rows * cols,
            "RUN_COUNTS": int(episode_seconds),
            "NUM_ROW": rows,
            "NUM_COL": cols,
            "TRAFFIC_FILE": route,
            "ROADNET_FILE": net,
            "SUMOCFG_FILE": sumocfg,
            "INTER_PHASE_MAPPING": mapping,
            "LIST_STATE_FEATURE": ["cur_phase", "time_this_phase", "traffic_movement_vehicle_ids_150m", "v9_cycle_150m_history"],
            "CAMERA_VIEW_DISTANCE": 150.0,
            "YELLOW_TIME": 5,
            "MIN_ACTION_TIME": 30,
            "INTERVAL": 1.0,
            "V9_TEMPORAL_FRAME_INTERVAL": 5,
            "SAVE_STATE_RNG": True,
            "SAVE_STATE_PRECISION": 15,
        }
        paths[city] = {"PATH_TO_DATA": str(data_dir), "PATH_TO_WORK_DIRECTORY": str(root / "runs" / "phase3_online" / city)}
    return configs, paths
