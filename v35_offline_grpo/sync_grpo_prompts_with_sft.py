"""Synchronize V35 GRPO prompts with the canonical reduced-pixel SFT prompt."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


STATIC_USER_MARKER = "Feature construction:"
OBSOLETE_TAGS = ("<current_v>", "</current_v>")


def _message_content(messages: Any, role: str) -> str:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    matches = [message for message in messages if isinstance(message, dict) and message.get("role") == role]
    if len(matches) != 1 or not isinstance(matches[0].get("content"), str):
        raise ValueError(f"expected exactly one {role!r} message")
    return matches[0]["content"]


def _canonical_prompt(sft_path: Path) -> tuple[str, str]:
    with sft_path.open(encoding="utf-8") as source:
        rows = json.load(source)
    if not isinstance(rows, list) or not rows:
        raise ValueError("SFT reference must be a non-empty JSON array")

    first_messages = rows[0].get("messages") if isinstance(rows[0], dict) else None
    canonical_system = _message_content(first_messages, "system")
    canonical_user = _message_content(first_messages, "user")
    marker_index = canonical_user.find(STATIC_USER_MARKER)
    if marker_index < 0:
        raise ValueError(f"SFT user prompt is missing {STATIC_USER_MARKER!r}")
    canonical_suffix = canonical_user[marker_index:]

    for line_number, row in enumerate(rows, 1):
        messages = row.get("messages") if isinstance(row, dict) else None
        if _message_content(messages, "system") != canonical_system:
            raise ValueError(f"SFT system prompt differs at record {line_number}")
        user = _message_content(messages, "user")
        row_marker_index = user.find(STATIC_USER_MARKER)
        if row_marker_index < 0 or user[row_marker_index:] != canonical_suffix:
            raise ValueError(f"SFT static user prompt differs at record {line_number}")

    for obsolete_tag in OBSOLETE_TAGS:
        if obsolete_tag in canonical_system or obsolete_tag in canonical_suffix:
            raise ValueError(f"SFT reference still contains obsolete top-level tag {obsolete_tag}")
    return canonical_system, canonical_suffix


def _synchronize_row(row: dict[str, Any], canonical_system: str, canonical_suffix: str) -> bool:
    messages = row.get("prompt")
    if not isinstance(messages, list):
        raise ValueError("GRPO row is missing prompt messages")

    system_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "system"]
    user_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "user"]
    if len(system_messages) != 1 or len(user_messages) != 1:
        raise ValueError("GRPO prompt must contain exactly one system and one user message")

    user_content = user_messages[0].get("content")
    if not isinstance(user_content, str):
        raise ValueError("GRPO user message content must be a string")
    marker_index = user_content.find(STATIC_USER_MARKER)
    if marker_index < 0:
        raise ValueError(f"GRPO user prompt is missing {STATIC_USER_MARKER!r}")

    synchronized_user = user_content[:marker_index] + canonical_suffix
    changed = system_messages[0].get("content") != canonical_system or user_content != synchronized_user
    system_messages[0]["content"] = canonical_system
    user_messages[0]["content"] = synchronized_user
    return changed


def _process(path: Path, canonical_system: str, canonical_suffix: str, in_place: bool) -> tuple[int, int]:
    total = 0
    changed = 0
    temporary_path: Path | None = None
    with path.open(encoding="utf-8") as source:
        destination = None
        if in_place:
            destination = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
            )
            temporary_path = Path(destination.name)
        try:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                try:
                    changed += _synchronize_row(row, canonical_system, canonical_suffix)
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                if destination is not None:
                    destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                total += 1
            if destination is not None:
                destination.flush()
                os.fsync(destination.fileno())
        except Exception:
            if destination is not None:
                destination.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        finally:
            if destination is not None and not destination.closed:
                destination.close()

    if in_place and temporary_path is not None:
        os.replace(temporary_path, path)
    return total, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-reference", required=True, type=Path)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    canonical_system, canonical_suffix = _canonical_prompt(args.sft_reference)
    for path in args.paths:
        total, changed = _process(path, canonical_system, canonical_suffix, args.in_place)
        print(json.dumps({"path": str(path), "records": total, "changed": changed, "in_place": args.in_place}))


if __name__ == "__main__":
    main()
