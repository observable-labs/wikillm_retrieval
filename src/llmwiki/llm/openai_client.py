"""OpenAI-compatible `/v1/chat/completions` adapter.

One adapter covers OpenAI, OpenRouter, Ollama, LM Studio, vLLM, and most
gateways — they all speak the same request shape. Streaming is used so long
generations don't sit behind a single socket read, and because it gives the
CLI live output for `ask`.
"""

from __future__ import annotations

from typing import Iterable

from ..config import LLMConfig
from ..errors import ProviderError
from .base import Completion, Message, TokenCallback
from ._http import post_sse


class OpenAICompatibleChatClient:
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._model = config.resolved_model()
        self._url = _chat_url(config.base_url or "https://api.openai.com/v1")
        self._headers = {"Authorization": f"Bearer {config.api_key}" if config.api_key else "", **config.extra_headers}

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: Iterable[Message],
        *,
        max_tokens: int,
        on_token: TokenCallback | None = None,
    ) -> Completion:
        payload: dict = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self._config.temperature is not None:
            payload["temperature"] = self._config.temperature

        parts: list[str] = []
        usage: dict = {}
        stop_reason: str | None = None
        for frame in post_sse(self._url, payload, self._headers, self._config.timeout):
            if frame.get("usage"):
                usage = frame["usage"]
            for choice in frame.get("choices") or []:
                delta = choice.get("delta") or {}
                chunk = delta.get("content")
                if chunk:
                    parts.append(chunk)
                    if on_token is not None:
                        on_token(chunk)
                if choice.get("finish_reason"):
                    stop_reason = choice["finish_reason"]

        text = "".join(parts)
        if not text.strip():
            raise ProviderError(
                f"{self._model} returned an empty response "
                "(the endpoint may not support streaming chat completions)"
            )
        return Completion(
            text=text,
            model=self._model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            stop_reason=stop_reason,
        )


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"
