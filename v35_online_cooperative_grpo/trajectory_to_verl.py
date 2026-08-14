"""Convert online trajectory JSONL into prompt/response GRPO records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(source: str, target: str) -> int:
    rows = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    output = Path(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {
                "data_source": "v35_online_cooperative_grpo",
                "prompt": [{"role": "user", "content": [
                    {"type": "text", "text": f"Intersection: {row['focal_id']}"},
                    *[{"type": "video", "video": path} for path in row.get("videos", {}).values()],
                ]}],
                "response": row["response"],
                "reward": float(row["reward"]),
                "advantage": float(row.get("advantage", 0.0)),
                "group_size": int(row.get("group_size", 1)),
                "snapshot_id": row["snapshot_id"],
                "branch_id": row["branch_id"],
                "signal": row["signal"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    print(json.dumps({"records": convert(args.source, args.target), "output": args.target}))


if __name__ == "__main__":
    main()
