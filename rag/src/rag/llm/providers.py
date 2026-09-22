"""Anthropic / OpenAI generation backends plus an offline echo stub.

Accessed over ``httpx`` for the same reason as the embedding providers: one
endpoint each, no vendor SDK, no LangChain wrapper.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from rag.config import settings
from rag.llm.base import LLMResponse, LLMService
from rag.logging_utils import get_logger
from rag.models import LLMUsage

logger = get_logger(__name__)

# USD per 1M tokens (input, output). Used only for cost estimation in the
# evaluation report; unknown models simply report cost as None.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (15.0, 75.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    for key, (pin, pout) in PRICING.items():
        if model.startswith(key):
            return round(input_tokens / 1e6 * pin + output_tokens / 1e6 * pout, 6)
    return None


class _HttpLLM(LLMService):
    def __init__(self, model: str, api_key: str, timeout: float = 120.0):
        if not api_key:
            raise RuntimeError(
                f"{type(self).__name__} requires an API key. Set it in rag/.env "
                "(see rag/.env.example)."
            )
        self.model = model
        self._api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    def _post(self, url: str, headers: dict, payload: dict, attempts: int = 3) -> Any:
        delay = 2.0
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:
                last = exc
                logger.warning(
                    "LLM request failed (%s), attempt %d/%d",
                    type(exc).__name__, attempt + 1, attempts,
                )
                if attempt < attempts - 1:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError("LLM request failed") from last


class AnthropicLLM(_HttpLLM):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(
            model or settings.LLM_MODEL,
            api_key or settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", ""),
        )

    def generate(self, system, prompt, max_tokens=1024, temperature=0.0) -> LLMResponse:
        data = self._post(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        text = "".join(
            block.get("text", "") for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        i, o = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        return LLMResponse(
            text=text.strip(),
            usage=LLMUsage(i, o, self.model, estimate_cost(self.model, i, o)),
        )


class OpenAILLM(_HttpLLM):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(
            model or settings.LLM_MODEL,
            api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", ""),
        )
        self._base = (settings.OPENAI_BASE_URL or "https://api.openai.com/v1").rstrip("/")

    def generate(self, system, prompt, max_tokens=1024, temperature=0.0) -> LLMResponse:
        data = self._post(
            f"{self._base}/chat/completions",
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        i, o = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        return LLMResponse(
            text=text.strip(),
            usage=LLMUsage(i, o, self.model, estimate_cost(self.model, i, o)),
        )


class EchoLLM(LLMService):
    """Offline stub. Emits an explicit insufficient-context style reply.

    Lets the API, CLI and pipeline wiring be exercised end-to-end with no
    credentials. It never fabricates facts.
    """

    def __init__(self, model: str = "echo-offline"):
        self.model = model

    def generate(self, system, prompt, max_tokens=1024, temperature=0.0) -> LLMResponse:
        return LLMResponse(
            text=(
                "[offline LLM stub] No generation backend is configured, so no "
                "grounded answer was produced. Set LLM_PROVIDER and the matching "
                "API key in rag/.env to enable generation."
            ),
            usage=LLMUsage(0, 0, self.model, 0.0),
        )


def build_llm_service(provider: str | None = None, model: str | None = None) -> LLMService:
    provider = (provider or settings.LLM_PROVIDER).lower()
    model = model or settings.LLM_MODEL
    if provider == "gemini":
        from rag.llm.gemini_provider import GeminiLLM

        return GeminiLLM(model=model)
    if provider == "anthropic":
        return AnthropicLLM(model=model)
    if provider == "openai":
        return OpenAILLM(model=model)
    if provider == "echo":
        return EchoLLM()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def build_judge_llm_service() -> LLMService:
    """The evaluator model - deliberately separate from the generator."""
    return build_llm_service(
        provider=settings.JUDGE_LLM_PROVIDER, model=settings.JUDGE_LLM_MODEL
    )
