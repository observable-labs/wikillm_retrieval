"""Anthropic Messages API adapter, via the official `anthropic` SDK.

Notes that matter for this pipeline:

* Streaming is always used. Step 2 of ingest can emit tens of thousands of
  tokens of FILE blocks, and non-streaming requests at that size hit HTTP
  timeouts.
* `temperature` is **not** sent. The current models reject sampling
  parameters with a 400, so the source project's `temperature: 0.1` has no
  equivalent here — determinism comes from the prompts instead.
* Adaptive thinking is on by default. Both ingest steps are exactly the kind
  of multi-constraint work it helps with.
"""

from __future__ import annotations

from typing import Iterable

from ..config import LLMConfig
from ..errors import ConfigError, ProviderError
from .base import Completion, Message, TokenCallback, split_system


class AnthropicChatClient:
    def __init__(self, config: LLMConfig) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - import guard
            raise ConfigError(
                "the Anthropic provider needs the SDK: pip install 'llmwiki[anthropic]'"
            ) from exc
        import anthropic

        self._anthropic = anthropic
        self._config = config
        self._model = config.resolved_model()

        kwargs: dict = {"timeout": config.timeout}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.extra_headers:
            kwargs["default_headers"] = dict(config.extra_headers)
        # A bare client also resolves `ant auth login` profiles and workload
        # identity, so an unset ANTHROPIC_API_KEY is not necessarily an error.
        self._client = anthropic.Anthropic(**kwargs)

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
        system, conversation = split_system(messages)
        payload: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in conversation],
        }
        if system:
            payload["system"] = system
        if self._config.thinking:
            payload["thinking"] = {"type": "adaptive"}
        if self._config.effort:
            payload["output_config"] = {"effort": self._config.effort}

        try:
            with self._client.messages.stream(**payload) as stream:
                if on_token is not None:
                    for chunk in stream.text_stream:
                        on_token(chunk)
                final = stream.get_final_message()
        except self._anthropic.APIStatusError as exc:
            raise ProviderError(_status_message(exc)) from exc
        except self._anthropic.APIConnectionError as exc:
            raise ProviderError(f"could not reach the Anthropic API: {exc}") from exc

        text = "".join(block.text for block in final.content if block.type == "text")
        # A refusal is a successful HTTP call with no usable output; surfacing
        # it as an error keeps the ingest queue from recording a bad page.
        if final.stop_reason == "refusal":
            details = getattr(final, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise ProviderError(f"the model declined this request (category: {category})")

        usage = getattr(final, "usage", None)
        return Completion(
            text=text,
            model=final.model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            stop_reason=final.stop_reason,
        )


def _status_message(exc) -> str:
    status = getattr(exc, "status_code", "?")
    if status == 401:
        return "Anthropic rejected the credentials (401). Check ANTHROPIC_API_KEY or run 'ant auth login'."
    if status == 429:
        return "Anthropic rate limit reached (429). Retry shortly or lower concurrency."
    return f"Anthropic API error {status}: {exc}"
