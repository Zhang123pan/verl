"""One-batch Ray/VLM/SUMO integration smoke test."""

from __future__ import annotations

import argparse
import base64
import json
import re
import urllib.request
from functools import partial
from pathlib import Path

import ray

from .city_env_config import build_city_env_configs
from .city_scheduler import load_sampling_config
from .ray_actors import BranchRequest, create_actor_classes
from .sumo_adapter import MultiCityBranchAdapter, SUMOEnvFactory, V30RecordingMasterFactory


SIGNALS = {"ETWT", "NTST", "ELWL", "NLSL"}


def _data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = "video/mp4" if suffix == ".mp4" else "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _load_prompt_template(path: str) -> tuple[str, str]:
    with open(path, encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    prompt = row.get("prompt") or row.get("messages")
    if not isinstance(prompt, list):
        raise ValueError(f"Prompt template has no prompt/messages list: {path}")
    system = next((m.get("content", "") for m in prompt if m.get("role") == "system"), "")
    user = next((m.get("content", "") for m in prompt if m.get("role") == "user"), "")
    if not system or not user:
        raise ValueError(f"Prompt template must contain system and user messages: {path}")
    return str(system), str(user)


def _query(api_url: str, api_key: str, model: str, videos: dict[str, str], timeout: float,
           temperature: float, prompt_template: tuple[str, str] | None, focal_id: str) -> str:
    if prompt_template:
        system_text, user_text = prompt_template
        user_text = re.sub(r"Intersection:\s*[^\n]+", f"Intersection: {focal_id}", user_text, count=1)
        user_text += "\n\nOnline smoke observation: no additional coordination frames are available for this cycle."
    else:
        system_text = "You are a visual traffic signal controller."
        user_text = (
            "Analyze this focal intersection using the four directional videos. "
            "Return only the required XML tags and choose one signal from ETWT, NTST, ELWL, NLSL."
        )
    content = [{"type": "text", "text": user_text}]
    for direction in ("E", "W", "N", "S"):
        content.append({"type": "text", "text": f"Direction {direction}:"})
        content.append({"type": "video_url", "video_url": {"url": _data_url(videos[direction])}})
    payload = {"model": model, "messages": [
                   {"role": "system", "content": system_text},
                   {"role": "user", "content": content}],
               "max_tokens": 2048, "temperature": temperature,
               "enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(api_url, data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    message = body["choices"][0]["message"]
    return str(message.get("content") or message.get("reasoning_content") or "")


def _signal(text: str) -> str | None:
    match = re.search(r"<signal>\s*(ETWT|NTST|ELWL|NLSL)\s*</signal>", text, re.I)
    return match.group(1).upper() if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("city_sampling.yaml")))
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--api-url", default="http://localhost:8088/v1/chat/completions")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-config", default="")
    parser.add_argument(
        "--prompt-template",
        default="/home/zhangpan/VLMTSCS/grpo_v30_offline_local_video_dataset_reduced_pixels/train.jsonl",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--cities", default="jinan")
    parser.add_argument("--rollout-n", type=int, default=6)
    parser.add_argument("--horizon-s", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--smoke-fallback",
        action="store_true",
        help="Use round-robin valid signals for invalid model output; smoke only.",
    )
    args = parser.parse_args()
    if args.api_config:
        private = json.loads(Path(args.api_config).read_text(encoding="utf-8"))
        args.api_url = private.get("api_url", args.api_url)
        args.api_key = private.get("api_key", args.api_key)
        args.model = private.get("model", args.model)
    if not args.model:
        parser.error("--model or api_config.model is required")
    prompt_template = _load_prompt_template(args.prompt_template)
    cfg = load_sampling_config(args.config)
    cities = [x.strip().lower() for x in args.cities.split(",") if x.strip()]
    configs, paths = build_city_env_configs(args.repo_root, cfg.episode_seconds)
    master_factory = V30RecordingMasterFactory(configs, paths, args.work_root, repo_root=args.repo_root)
    branch_factory = SUMOEnvFactory(configs, paths, args.work_root, repo_root=args.repo_root)
    ray.init(ignore_reinit_error=True)
    _Master, Branch, RotatingMaster = create_actor_classes()
    seed_map = {spec.name: list(spec.seeds) for spec in cfg.cities}
    masters = {city: RotatingMaster.remote(master_factory, city, seed_map[city],
                                           str(Path(args.snapshot_dir) / city), cfg.episode_seconds)
               for city in cities}
    branches = []
    try:
        snapshots = ray.get([masters[city].advance_and_publish.remote(cfg.control_period_seconds)
                             for city in cities])
        all_results = []
        for snapshot in snapshots:
            if not snapshot.observations:
                raise RuntimeError("Snapshot has no V30 observations")
            focal_id, observation = sorted(snapshot.observations.items())[0]
            for branch_id in range(args.rollout_n):
                raw = _query(args.api_url, args.api_key, args.model, observation.videos,
                             args.timeout, args.temperature, prompt_template, focal_id)
                signal = _signal(raw)
                fallback = signal is None
                if fallback:
                    if not args.smoke_fallback:
                        raise RuntimeError(
                            f"Invalid VLM signal for branch {branch_id}: {raw[:500]!r}"
                        )
                    signal = ("ETWT", "NTST", "ELWL", "NLSL")[branch_id % 4]
                actions = {inter_id: "ETWT" for inter_id in configs[snapshot.city]["INTER_PHASE_MAPPING"]}
                actions[focal_id] = signal
                print(json.dumps({"snapshot_id": snapshot.snapshot_id, "focal_id": focal_id,
                                  "branch_id": branch_id, "signal": signal,
                                  "fallback": fallback, "response": raw},
                                 ensure_ascii=False))
                actor = Branch.remote(partial(MultiCityBranchAdapter, branch_factory, f"vlm_{len(branches)}"), len(branches))
                branches.append(actor)
                all_results.append(actor.rollout.remote(
                    BranchRequest(snapshot, actions, reward_region={"focal_id": focal_id}, horizon_s=args.horizon_s),
                    branch_id))
        results = ray.get(all_results)
        print(json.dumps({"groups": {s.snapshot_id: sum(r["snapshot_id"] == s.snapshot_id for r in results)
                                      for s in snapshots},
                          "rewards": [r["reward"] for r in results]}, ensure_ascii=False))
    finally:
        if branches:
            try:
                ray.get([actor.close.remote() for actor in branches])
            except ray.exceptions.RayError:
                pass
        try:
            ray.get([actor.close.remote() for actor in masters.values()])
        except ray.exceptions.RayError:
            pass
        ray.shutdown()


if __name__ == "__main__":
    main()
