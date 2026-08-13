"""Small OpenAI-compatible vLLM policy adapter for online rollouts."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

from .episode import GeneratedResponse


@dataclass
class OpenAIChatPolicy:
    api_url: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_s: float = 900.0

    def generate(self, messages: list[dict[str, Any]], *, branch_id: int, role: str) -> GeneratedResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Invalid vLLM response: {body!r}") from exc
        return GeneratedResponse(
            text=str(text),
            rollout_metadata={"branch_id": branch_id, "role": role, "model": self.model},
        )
