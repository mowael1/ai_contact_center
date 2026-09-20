"""Google Gemini 2.5 generation backend.

Talks to the Generative Language REST API directly over ``httpx`` - one
endpoint for blocking calls and one SSE endpoint for streaming - so no vendor
SDK and no LangChain wrapper is needed.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Iterator

import httpx

from rag.config import settings
from rag.llm.base import LLMResponse, LLMService, StreamChunk
from rag.logging_utils import get_logger
from rag.models import LLMUsage

logger = get_logger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

#: Free-tier Gemini allows roughly 15 requests/minute. The agentic loop fires
#: several calls per question, so without pacing an evaluation run trips 429s
#: almost immediately. 0 disables pacing.
_RATE_LIMIT_STATUS = 429


class _RateLimiter:
    """Process-wide minimum interval between Gemini requests."""

    def __init__(self, rpm: int) -> None:
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if not self._min_interval:
            return
        with self._lock:
            delta = time.monotonic() - self._last
            if delta < self._min_interval:
                time.sleep(self._min_interval - delta)
            self._last = time.monotonic()

# USD per 1M tokens (input, output).
GEMINI_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}


def gemini_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    for key, (pin, pout) in GEMINI_PRICING.items():
        if model.startswith(key):
            return round(input_tokens / 1e6 * pin + output_tokens / 1e6 * pout, 6)
    return None


class GeminiLLM(LLMService):
    """Gemini 2.5 (``gemini-2.5-pro`` / ``gemini-2.5-flash``)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.model = model or settings.LLM_MODEL
        self._api_key = (
            api_key
            or settings.GOOGLE_API_KEY
            or os.getenv("GOOGLE_API_KEY", "")
            or os.getenv("GEMINI_API_KEY", "")
        )
        if not self._api_key:
            raise RuntimeError(
                "GeminiLLM requires an API key. Set GOOGLE_API_KEY in rag/.env "
                "(see rag/.env.example). No credential is ever read from code."
            )
        self._client = httpx.Client(timeout=timeout)
        self._limiter = _RateLimiter(settings.LLM_REQUESTS_PER_MINUTE)

    # -- request construction ----------------------------------------------
    def _payload(self, system: str, prompt: str, max_tokens: int, temperature: float) -> dict:
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                # Gemini 2.5 "thinks" by default; the budget is disabled so
                # reasoning tokens are not billed for a grounded extraction task.
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

    @property
    def _headers(self) -> dict[str, str]:
        # The key travels in a header, never in the URL, so it cannot leak
        # through request logs or error messages that echo the URL.
        return {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}

    def _usage(self, data: dict[str, Any]) -> LLMUsage:
        meta = data.get("usageMetadata") or {}
        prompt_tokens = int(meta.get("promptTokenCount", 0))
        output_tokens = int(
            meta.get("candidatesTokenCount", 0) + meta.get("thoughtsTokenCount", 0)
        )
        return LLMUsage(
            prompt_tokens, output_tokens, self.model,
            gemini_cost(self.model, prompt_tokens, output_tokens),
        )

    @staticmethod
    def _text_of(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts)

    # -- blocking -----------------------------------------------------------
    def generate(self, system, prompt, max_tokens=1024, temperature=0.0) -> LLMResponse:
        url = f"{API_ROOT}/{self.model}:generateContent"
        payload = self._payload(system, prompt, max_tokens, temperature)
        attempts = settings.LLM_MAX_RETRIES
        delay = 2.0
        last: Exception | None = None

        for attempt in range(attempts):
            self._limiter.wait()
            try:
                response = self._client.post(url, headers=self._headers, json=payload)
                if response.status_code == _RATE_LIMIT_STATUS:
                    wait_for = _retry_after(response, fallback=delay * 4)
                    logger.warning(
                        "Gemini rate-limited, waiting %.0fs (attempt %d/%d)",
                        wait_for, attempt + 1, attempts,
                    )
                    time.sleep(wait_for)
                    last = RuntimeError("rate limited")
                    continue
                response.raise_for_status()
                data = response.json()
                return LLMResponse(self._text_of(data).strip(), self._usage(data))
            except httpx.HTTPError as exc:
                last = exc
                logger.warning(
                    "Gemini request failed (%s), attempt %d/%d",
                    type(exc).__name__, attempt + 1, attempts,
                )
                if attempt < attempts - 1:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError(f"Gemini request failed after {attempts} attempts") from last

    # -- streaming ----------------------------------------------------------
    def generate_stream(
        self, system, prompt, max_tokens=1024, temperature=0.0
    ) -> Iterator[StreamChunk]:
        """Stream via ``:streamGenerateContent?alt=sse``.

        Yields a chunk per delta, then a final ``done`` chunk carrying the full
        text and the usage metadata Gemini reports on the last event.
        """
        url = f"{API_ROOT}/{self.model}:streamGenerateContent"
        payload = self._payload(system, prompt, max_tokens, temperature)
        accumulated: list[str] = []
        usage = LLMUsage(0, 0, self.model, 0.0)

        self._limiter.wait()
        with self._client.stream(
            "POST", url, headers=self._headers, json=payload, params={"alt": "sse"}
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug("Skipping unparseable SSE frame")
                    continue

                delta = self._text_of(data)
                if data.get("usageMetadata"):
                    usage = self._usage(data)
                if delta:
                    accumulated.append(delta)
                    yield StreamChunk(delta=delta, text="".join(accumulated))

        full = "".join(accumulated).strip()
        yield StreamChunk(delta="", text=full, done=True, usage=usage)


def _retry_after(response: httpx.Response, fallback: float) -> float:
    """Honour a Retry-After header when the API sends one."""
    raw = response.headers.get("retry-after")
    if raw:
        try:
            return min(120.0, max(1.0, float(raw)))
        except ValueError:
            pass
    # Google also returns the delay inside the error body.
    try:
        for detail in response.json().get("error", {}).get("details", []):
            delay = detail.get("retryDelay")
            if delay and delay.endswith("s"):
                return min(120.0, max(1.0, float(delay[:-1])))
    except Exception:
        pass
    return min(120.0, fallback)
