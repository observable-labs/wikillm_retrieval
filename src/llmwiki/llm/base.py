"""The provider-neutral chat interface the pipeline codes against.

Deliberately small: the ingest and query stages need exactly one operation —
"send these messages, give me the whole text back" — plus an optional token
callback so the CLI can stream an answer as it arrives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

TokenCallback = Callable[[str], None]


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None


class ChatClient(Protocol):
    """What `ingest` and `query` require of a provider."""

    def complete(
        self,
        messages: Iterable[Message],
        *,
        max_tokens: int,
        on_token: TokenCallback | None = None,
    ) -> Completion:
        ...


def split_system(messages: Iterable[Message]) -> tuple[str, list[Message]]:
    """Anthropic takes `system` as a top-level parameter, not a message."""
    system_parts: list[str] = []
    rest: list[Message] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
        else:
            rest.append(message)
    return "\n\n".join(part for part in system_parts if part.strip()), rest
