"""Explicit Ray/SUMO smoke launcher; it does not run during import or tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import ray

from .city_env_config import build_city_env_configs
from .city_scheduler import CitySampler, load_sampling_config
from .ray_actors import create_actor_classes
from .sumo_adapter import SUMOEnvFactory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("city_sampling.yaml")))
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--batches", type=int, default=1)
    args = parser.parse_args()
    cfg = load_sampling_config(args.config)
    configs, paths = build_city_env_configs(args.repo_root, cfg.episode_seconds)
    factory = SUMOEnvFactory(configs, paths, args.work_root, repo_root=args.repo_root)
    ray.init(ignore_reinit_error=True)
    _Master, _Branch, RotatingMaster = create_actor_classes()
    masters = {
        city.name: RotatingMaster.remote(factory, city.name, list(city.seeds), str(Path(args.snapshot_dir) / city.name), cfg.episode_seconds)
        for city in cfg.cities
    }
    sampler = CitySampler(cfg, seed=0)
    for batch_index in range(args.batches):
        cities = sampler.next_batch_cities()
        snapshots = ray.get([masters[city].advance_and_publish.remote(cfg.control_period_seconds) for city in cities])
        print({"batch": batch_index, "cities": cities, "snapshots": [snapshot.snapshot_id for snapshot in snapshots]})
    ray.get([actor.close.remote() for actor in masters.values()])


if __name__ == "__main__":
    main()
