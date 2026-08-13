"""Ray-managed counterfactual branch smoke for one mixed-city batch.

This intentionally uses fixed phase actions and a zero placeholder reward.  Its
purpose is to validate that every GRPO group starts from the same master
snapshot and that each branch advances its own SUMO process independently.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import ray

from .city_env_config import build_city_env_configs
from .city_scheduler import load_sampling_config, CitySampler
from .ray_actors import BranchRequest, create_actor_classes
from .sumo_adapter import MultiCityBranchAdapter, SUMOEnvFactory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("city_sampling.yaml")))
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--rollout-n", type=int, default=None)
    args = parser.parse_args()
    cfg = load_sampling_config(args.config)
    rollout_n = int(args.rollout_n or cfg.rollout_n)
    configs, paths = build_city_env_configs(args.repo_root, cfg.episode_seconds)
    factory = SUMOEnvFactory(configs, paths, args.work_root, repo_root=args.repo_root)

    ray.init(ignore_reinit_error=True)
    _Master, Branch, RotatingMaster = create_actor_classes()
    masters = {
        city.name: RotatingMaster.remote(
            factory, city.name, list(city.seeds), str(Path(args.snapshot_dir) / city.name), cfg.episode_seconds
        )
        for city in cfg.cities
    }
    sampler = CitySampler(cfg, seed=0)
    branches = []
    try:
        for batch_index in range(args.batches):
            cities = sampler.next_batch_cities()
            snapshots = ray.get([
                masters[city].advance_and_publish.remote(cfg.control_period_seconds)
                for city in cities
            ])
            requests = []
            for snapshot in snapshots:
                # All intersections receive one valid canonical phase. Real
                # policy actions replace this table after vLLM integration.
                actions = {inter_id: "ETWT" for inter_id in configs[snapshot.city]["INTER_PHASE_MAPPING"]}
                for branch_id in range(rollout_n):
                    requests.append((snapshot, actions, branch_id))

            branches = [
                Branch.remote(
                    partial(MultiCityBranchAdapter, factory, str(index)), index
                )
                for index in range(len(requests))
            ]
            refs = [
                actor.rollout.remote(
                    BranchRequest(snapshot, actions, reward_region={}, horizon_s=cfg.control_period_seconds),
                    branch_id,
                )
                for actor, (snapshot, actions, branch_id) in zip(branches, requests)
            ]
            results = ray.get(refs)
            grouped = {}
            for result in results:
                grouped.setdefault(result["snapshot_id"], []).append(result)
            print({
                "batch": batch_index,
                "groups": {key: len(value) for key, value in grouped.items()},
                "sim_times": sorted({result["environment_result"]["sim_time_s"] for result in results}),
            })
    finally:
        if branches:
            ray.get([actor.close.remote() for actor in branches])
        ray.get([actor.close.remote() for actor in masters.values()])
        ray.shutdown()


if __name__ == "__main__":
    main()
