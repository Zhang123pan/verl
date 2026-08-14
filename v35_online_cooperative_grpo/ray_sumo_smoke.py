"""Explicit Ray/SUMO smoke launcher; it does not run during import or tests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .city_env_config import build_city_env_configs
from .city_scheduler import CitySampler, load_sampling_config
from .ray_actors import create_actor_classes
from .sumo_adapter import SUMOEnvFactory, V30RecordingMasterFactory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("city_sampling.yaml")))
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--record-observations", action="store_true")
    parser.add_argument("--cities", default="", help="Optional comma-separated city subset for recording diagnostics.")
    parser.add_argument("--renderer-gpus-per-master", type=float, default=1.0)
    args = parser.parse_args()
    cfg = load_sampling_config(args.config)
    configs, paths = build_city_env_configs(args.repo_root, cfg.episode_seconds)
    factory_class = V30RecordingMasterFactory if args.record_observations else SUMOEnvFactory
    factory = factory_class(configs, paths, args.work_root, repo_root=args.repo_root)
    # p3headlessgl/EGL enumerates physical render devices.  Ray's default
    # CUDA_VISIBLE_DEVICES remapping makes an otherwise working run_v35
    # renderer fail inside an actor, even though Ray has reserved a GPU for
    # that actor.  Keep Ray's num_gpus placement/accounting, but preserve the
    # driver's physical device visibility for the renderer process.
    if args.record_observations:
        os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    # Import Ray only after the visibility policy is fixed.  Ray caches parts
    # of its accelerator setup during import, so setting this afterwards is
    # too late for p3headlessgl actors.
    import ray
    ray.init(ignore_reinit_error=True)
    _Master, _Branch, RotatingMaster = create_actor_classes()
    selected_names = [value.strip() for value in args.cities.split(",") if value.strip()]
    selected = [city for city in cfg.cities if not selected_names or city.name in selected_names]
    unknown = set(selected_names) - {city.name for city in cfg.cities}
    if unknown:
        raise ValueError(f"Unknown cities: {sorted(unknown)}")
    if not selected:
        raise ValueError("At least one city must be selected.")
    actor_options = {"num_gpus": args.renderer_gpus_per_master} if args.record_observations else {}
    masters = {
        city.name: RotatingMaster.options(**actor_options).remote(
            factory, city.name, list(city.seeds), str(Path(args.snapshot_dir) / city.name), cfg.episode_seconds
        )
        for city in selected
    }
    sampler = CitySampler(cfg, seed=0)
    for batch_index in range(args.batches):
        cities = sampler.next_batch_cities() if not selected_names else [city.name for city in selected]
        snapshots = ray.get([masters[city].advance_and_publish.remote(cfg.control_period_seconds) for city in cities])
        print({
            "batch": batch_index,
            "cities": cities,
            "snapshots": [snapshot.snapshot_id for snapshot in snapshots],
            "observation_counts": [len(snapshot.observations or {}) for snapshot in snapshots],
        })
    ray.get([actor.close.remote() for actor in masters.values()])


if __name__ == "__main__":
    main()
