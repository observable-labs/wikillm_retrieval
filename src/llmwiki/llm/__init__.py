"""Provider adapters and the factory that picks between them."""

from __future__ import annotations

from ..config import ANTHROPIC, OPENAI, LLMConfig
from ..errors import ConfigError
from .base import ChatClient, Completion, Message, TokenCallback

__all__ = ["ChatClient", "Completion", "Message", "TokenCallback", "build_client"]


def build_client(config: LLMConfig) -> ChatClient:
    if config.provider == ANTHROPIC:
        from .anthropic_client import AnthropicChatClient

        return AnthropicChatClient(config)
    if config.provider == OPENAI:
        from .openai_client import OpenAICompatibleChatClient

        return OpenAICompatibleChatClient(config)
    raise ConfigError(f"unknown provider {config.provider!r}")
