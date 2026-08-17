"""Expand cooperative branch trajectories into policy-training spans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


POLICY_FIELDS = ("prompt_ids", "response_ids", "response_logprobs", "response_mask")


def _policy_metadata(span: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    metadata = span.get("policy_metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("policy_metadata must be an object")
    values = {field: metadata.get(field, span.get(field)) for field in POLICY_FIELDS}
    present = [values[field] is not None for field in POLICY_FIELDS]
    if any(present) and not all(present):
        missing = [field for field, exists in zip(POLICY_FIELDS, present) if not exists]
        raise ValueError(f"partial policy metadata; missing {missing}")
    if not all(present):
        return {}, False
    prompt_ids = list(values["prompt_ids"])
    response_ids = list(values["response_ids"])
    response_logprobs = list(values["response_logprobs"])
    response_mask = list(values["response_mask"])
    if not prompt_ids or not response_ids:
        raise ValueError("policy token sequences must be non-empty")
    if len(response_ids) != len(response_logprobs) or len(response_ids) != len(response_mask):
        raise ValueError("response_ids, response_logprobs, and response_mask must have equal lengths")
    if any(value not in (0, 1) for value in response_mask):
        raise ValueError("response_mask values must be 0 or 1")
    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "response_logprobs": response_logprobs,
        "response_mask": response_mask,
    }, True


def _span_record(branch: dict[str, Any], span: dict[str, Any], role: str, index: int) -> dict[str, Any]:
    snapshot_id = str(branch["snapshot_id"])
    branch_id = int(branch["branch_id"])
    intersection_id = str(span["intersection_id"] if role == "receiver" else branch["focal_id"])
    prompt = span.get("prompt")
    response = span.get("response")
    videos = span.get("videos") or {}
    if not isinstance(prompt, list) or not isinstance(response, str) or not response:
        raise ValueError(f"{role} span must contain its complete prompt and non-empty response")
    if not isinstance(videos, dict) or set(videos) != {"E", "W", "N", "S"}:
        raise ValueError(f"{role} span must contain E/W/N/S videos")
    policy_metadata, policy_ready = _policy_metadata(span)
    record = {
        "data_source": "v35_online_cooperative_grpo",
        "uid": snapshot_id,
        "snapshot_id": snapshot_id,
        "branch_id": branch_id,
        "branch_uid": f"{snapshot_id}:branch:{branch_id}",
        "span_uid": f"{snapshot_id}:branch:{branch_id}:{role}:{index}",
        "role": role,
        "intersection_id": intersection_id,
        "prompt": prompt,
        "response": response,
        "videos": [videos[direction] for direction in ("E", "W", "N", "S")],
        "reward": float(branch["reward"]),
        "advantage": float(branch["advantage"]),
        "group_size": int(branch["group_size"]),
        "signal": str(span.get("signal", branch.get("signal", ""))),
        "policy_ready": policy_ready,
    }
    record.update(policy_metadata)
    return record


def expand_branch(branch: dict[str, Any]) -> list[dict[str, Any]]:
    """Assign one branch advantage to its sender and all receiver spans."""
    required = ("snapshot_id", "branch_id", "focal_id", "reward", "advantage", "group_size")
    missing = [field for field in required if field not in branch]
    if missing:
        raise ValueError(f"branch is missing {missing}")
    sender = {
        "prompt": branch.get("prompt"),
        "response": branch.get("response"),
        "videos": branch.get("videos"),
        "signal": branch.get("signal"),
        "policy_metadata": branch.get("policy_metadata"),
    }
    records = [_span_record(branch, sender, "sender", 0)]
    receivers = branch.get("receivers") or []
    if not isinstance(receivers, list):
        raise ValueError("receivers must be a list")
    records.extend(_span_record(branch, receiver, "receiver", index)
                   for index, receiver in enumerate(receivers))
    return records


def convert_rows(rows: Iterable[dict[str, Any]], require_policy_metadata: bool = False) -> list[dict[str, Any]]:
    records = [record for branch in rows for record in expand_branch(branch)]
    if require_policy_metadata:
        missing = [record["span_uid"] for record in records if not record["policy_ready"]]
        if missing:
            raise ValueError(f"{len(missing)} spans lack on-policy token/logprob metadata; first={missing[0]}")
    return records


def convert(source: str, target: str, require_policy_metadata: bool = False) -> int:
    rows = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    records = convert_rows(rows, require_policy_metadata=require_policy_metadata)
    output = Path(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--require-policy-metadata", action="store_true")
    args = parser.parse_args()
    count = convert(args.source, args.target, require_policy_metadata=args.require_policy_metadata)
    print(json.dumps({"records": count, "output": args.target}, ensure_ascii=False))


if __name__ == "__main__":
    main()
