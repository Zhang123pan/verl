"""One-batch Ray/VLM/SUMO integration smoke test."""

from __future__ import annotations

import argparse
import base64
import json
import re
import urllib.request
from copy import deepcopy
from functools import partial
from pathlib import Path

import ray

from .city_env_config import build_city_env_configs
from .city_scheduler import load_sampling_config
from .ray_actors import BranchRequest, create_actor_classes
from .decision_parser import resolve_decision
from .message_router import (
    MessageProtocolError,
    parse_sender_message,
    render_receiver_message_context,
    route_messages,
    select_sender,
)
from .prompt_builder import build_receiver_prompt, build_sender_prompt
from .sumo_adapter import MultiCityBranchAdapter, SUMOEnvFactory, V30RecordingMasterFactory
from .trajectory import summarize_smoke, write_group_batch


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


def _base_messages(prompt_template: tuple[str, str], intersection_id: str) -> list[dict[str, str]]:
    system_text, user_text = prompt_template
    user_text = re.sub(r"Intersection:\s*[^\n]+", f"Intersection: {intersection_id}", user_text, count=1)
    user_text += "\n\nOnline smoke observation: no additional coordination frames are available for this cycle."
    return [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]


def _query(api_url: str, api_key: str, model: str, videos: dict[str, str], timeout: float,
           temperature: float, messages: list[dict[str, object]]) -> str:
    messages = deepcopy(messages)
    users = [message for message in messages if message.get("role") == "user"]
    if len(users) != 1 or not isinstance(users[0].get("content"), str):
        raise ValueError("Cooperative VLM query requires exactly one string user message")
    user_text = str(users[0]["content"])
    content = [{"type": "text", "text": user_text}]
    for direction in ("E", "W", "N", "S"):
        content.append({"type": "text", "text": f"Direction {direction}:"})
        content.append({"type": "video_url", "video_url": {"url": _data_url(videos[direction])}})
    users[0]["content"] = content
    payload = {"model": model, "messages": messages,
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


def _route_table(city: str) -> dict:
    path = Path(__file__).with_name("artifacts") / f"movement_routes_{city}.json"
    return json.loads(path.read_text(encoding="utf-8"))


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
    parser.add_argument("--focal-id", default="")
    parser.add_argument("--rollout-n", type=int, default=6)
    parser.add_argument("--horizon-s", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--trajectory-output", default="")
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
        trajectory_records = []
        for snapshot in snapshots:
            if not snapshot.observations:
                raise RuntimeError("Snapshot has no V30 observations")
            route_table = _route_table(snapshot.city)
            focal_id = select_sender(set(snapshot.observations), route_table, args.focal_id)
            observation = snapshot.observations[focal_id]
            if not snapshot.background_actions:
                raise RuntimeError(f"Snapshot {snapshot.snapshot_id} has no V25 background actions")
            for branch_id in range(args.rollout_n):
                sender_prompt = build_sender_prompt(_base_messages(prompt_template, focal_id))
                raw = _query(args.api_url, args.api_key, args.model, observation.videos,
                             args.timeout, args.temperature, sender_prompt)
                v25_signal = snapshot.background_actions.get(focal_id)
                decision = resolve_decision(raw, v25_signal)
                signal = decision.signal
                sender_signal_valid = decision.signal_source == "model_signal"
                sender_message_valid = False
                fallback = decision.signal_source == "v25_fallback"
                routed = []
                try:
                    parsed_messages = parse_sender_message(raw, selected_signal=signal)
                    routed = route_messages(focal_id, parsed_messages, route_table)
                    sender_message_valid = True
                except (MessageProtocolError, KeyError, ValueError):
                    # A message is coordination-only: retain a valid selected signal
                    # and simply give receivers no model override this cycle.
                    pass
                actions = dict(snapshot.background_actions)
                actions[focal_id] = signal
                receiver_contexts = render_receiver_message_context(routed)
                receiver_records = []
                for receiver_id, message_context in sorted(receiver_contexts.items()):
                    receiver_observation = snapshot.observations.get(receiver_id)
                    if receiver_observation is None:
                        raise RuntimeError(f"Snapshot has no observation for receiver {receiver_id}")
                    receiver_prompt = build_receiver_prompt(
                        _base_messages(prompt_template, receiver_id), message_context
                    )
                    receiver_raw = _query(
                        args.api_url, args.api_key, args.model, receiver_observation.videos,
                        args.timeout, args.temperature, receiver_prompt,
                    )
                    receiver_decision = resolve_decision(
                        receiver_raw, snapshot.background_actions[receiver_id]
                    )
                    receiver_signal = receiver_decision.signal
                    receiver_fallback = receiver_decision.signal_source == "v25_fallback"
                    fallback = fallback or receiver_fallback
                    actions[receiver_id] = receiver_signal
                    receiver_records.append({
                        "intersection_id": receiver_id,
                        "signal": receiver_signal,
                        "fallback": receiver_fallback,
                        "perception_valid": receiver_decision.perception_valid,
                        "perception_error": receiver_decision.perception_error,
                        "response": receiver_raw,
                        "prompt": receiver_prompt,
                        "videos": dict(receiver_observation.videos),
                        "message_context": message_context,
                    })
                receiver_ids = tuple(sorted(receiver_contexts))
                print(json.dumps({"snapshot_id": snapshot.snapshot_id, "focal_id": focal_id,
                                  "branch_id": branch_id, "signal": signal,
                                  "receiver_ids": receiver_ids,
                                  "fallback": fallback, "response": raw},
                                 ensure_ascii=False))
                actor = Branch.remote(partial(MultiCityBranchAdapter, branch_factory, f"vlm_{len(branches)}"), len(branches))
                branches.append(actor)
                all_results.append(actor.rollout.remote(
                    BranchRequest(
                        snapshot,
                        actions,
                        reward_region={"focal_id": focal_id, "receiver_ids": receiver_ids},
                        horizon_s=args.horizon_s,
                    ),
                    branch_id))
                trajectory_records.append({
                    "snapshot_id": snapshot.snapshot_id,
                    "city": snapshot.city,
                    "focal_id": focal_id,
                    "branch_id": branch_id,
                    "signal": signal,
                    "sender_signal_valid": sender_signal_valid,
                    "sender_message_valid": sender_message_valid,
                    "perception_valid": decision.perception_valid,
                    "perception_error": decision.perception_error,
                    "fallback": fallback,
                    "response": raw,
                    "prompt": sender_prompt,
                    "receivers": receiver_records,
                    "receiver_ids": receiver_ids,
                    "routed_messages": [item.__dict__ for item in routed],
                    "videos": dict(observation.videos),
                    "prompt_template": args.prompt_template,
                    "model": args.model,
                    "actions": actions,
                })
        results = ray.get(all_results)
        for record, result in zip(trajectory_records, results):
            record["reward"] = float(result["reward"])
            record["environment_result"] = result.get("environment_result", {})
        summary = summarize_smoke(trajectory_records)
        if args.trajectory_output:
            write_group_batch(args.trajectory_output, trajectory_records)
            print(json.dumps({"trajectory_output": args.trajectory_output,
                              "trajectory_records": len(trajectory_records)}, ensure_ascii=False))
        print(json.dumps({"groups": {s.snapshot_id: sum(r["snapshot_id"] == s.snapshot_id for r in results)
                                      for s in snapshots},
                          "rewards": [r["reward"] for r in results],
                          "smoke_summary": summary}, ensure_ascii=False))
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
