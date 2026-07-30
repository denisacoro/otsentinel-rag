from __future__ import annotations

import time

import httpx


class OllamaClient:
    """Minimal client for a local Ollama chat model."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, float]:
        """Send messages to Ollama and return (answer_text, latency_ms)."""

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        start = time.perf_counter()

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        latency_ms = (time.perf_counter() - start) * 1000
        data = response.json()

        answer_text = str(data["message"]["content"]).strip()

        return answer_text, latency_ms