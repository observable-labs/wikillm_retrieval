"""The OpenAI-compatible adapters, driven against a real local HTTP server.

Stub clients verify the pipeline; these verify the wire — SSE accumulation,
error surfacing, URL derivation, and the embeddings batch contract.
"""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import pytest

from llmwiki.config import EmbeddingConfig, LLMConfig
from llmwiki.embeddings import TRANSPORT_ATTEMPTS, embed_texts
from llmwiki.errors import ProviderError, ProviderTransportError
from llmwiki.llm._http import _error_detail, post_json
from llmwiki.llm.base import Message
from llmwiki.llm.openai_client import OpenAICompatibleChatClient, _chat_url


class _Handler(BaseHTTPRequestHandler):
    script: dict = {}
    received: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).received.append((self.path, body, dict(self.headers)))
        status, frames = type(self).script.get(self.path, (404, []))

        if status != 200:
            payload = json.dumps({"error": {"message": "model not found"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path.endswith("/embeddings"):
            payload = json.dumps(
                {"data": [{"index": i, "embedding": [float(i), 1.0]} for i in range(len(body["input"]))]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for frame in frames:
            self.wfile.write(f"data: {frame}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.script = {}
    _Handler.received = []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, _Handler
    httpd.shutdown()


def _base_url(httpd) -> str:
    return f"http://127.0.0.1:{httpd.server_port}/v1"


def test_sse_deltas_are_accumulated_and_streamed(server):
    httpd, handler = server
    handler.script["/v1/chat/completions"] = (
        200,
        [
            json.dumps({"choices": [{"delta": {"content": "Hello "}}]}),
            json.dumps({"choices": [{"delta": {"content": "world"}}]}),
            "{ this frame is malformed",  # a bad frame must not abort the stream
            json.dumps({"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 3}}),
        ],
    )
    client = OpenAICompatibleChatClient(
        LLMConfig(provider="openai", model="m", base_url=_base_url(httpd), api_key="k")
    )
    streamed: list[str] = []
    completion = client.complete(
        [Message("system", "s"), Message("user", "u")], max_tokens=64, on_token=streamed.append
    )

    assert completion.text == "Hello world!"
    assert "".join(streamed) == "Hello world!"
    assert completion.output_tokens == 3
    assert completion.stop_reason == "stop"

    _path, body, headers = handler.received[0]
    assert body["stream"] is True
    assert body["messages"][0] == {"role": "system", "content": "s"}
    assert headers["Authorization"] == "Bearer k"


def test_http_error_body_is_surfaced(server):
    httpd, handler = server
    handler.script["/v1/chat/completions"] = (404, [])
    client = OpenAICompatibleChatClient(
        LLMConfig(provider="openai", model="nope", base_url=_base_url(httpd))
    )
    with pytest.raises(ProviderError, match="model not found"):
        client.complete([Message("user", "u")], max_tokens=16)


def test_empty_stream_is_an_error(server):
    httpd, handler = server
    handler.script["/v1/chat/completions"] = (200, [])
    client = OpenAICompatibleChatClient(
        LLMConfig(provider="openai", model="m", base_url=_base_url(httpd))
    )
    with pytest.raises(ProviderError, match="empty response"):
        client.complete([Message("user", "u")], max_tokens=16)


def test_unreachable_endpoint_is_a_provider_error():
    client = OpenAICompatibleChatClient(
        LLMConfig(provider="openai", model="m", base_url="http://127.0.0.1:1/v1", timeout=2)
    )
    with pytest.raises(ProviderError, match="could not reach"):
        client.complete([Message("user", "u")], max_tokens=16)


@pytest.mark.parametrize(
    "base,expected",
    [
        ("https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"),
        ("http://localhost:11434/v1/", "http://localhost:11434/v1/chat/completions"),
        ("http://host:8000", "http://host:8000/v1/chat/completions"),
        ("https://gw/x/chat/completions", "https://gw/x/chat/completions"),
    ],
)
def test_chat_url_derivation(base, expected):
    assert _chat_url(base) == expected


def test_embeddings_batch_and_ordering(server):
    httpd, handler = server
    handler.script["/v1/embeddings"] = (200, [])
    config = EmbeddingConfig(
        enabled=True, model="e", base_url=_base_url(httpd), api_key="k", batch_size=2
    )
    vectors = embed_texts(["a", "b", "c"], config)

    assert len(vectors) == 3
    # Two requests: batch_size 2 over 3 inputs.
    assert len(handler.received) == 2
    assert handler.received[0][1]["input"] == ["a", "b"]
    assert handler.received[1][1]["input"] == ["c"]


def test_embeddings_require_configuration():
    with pytest.raises(Exception, match="not configured"):
        embed_texts(["a"], EmbeddingConfig())


@pytest.mark.parametrize(
    "body, expected",
    [
        ('{"error": {"message": "credit balance is too low"}}', "credit balance is too low"),
        ('{"error": "flat string"}', "flat string"),
        # Gemini's OpenAI-compatible surface answers an unauthenticated request
        # with a top-level list. `.get` on it raised AttributeError, so the
        # error path crashed with a traceback instead of reporting the error.
        ('[{"error": {"message": "API key not valid", "code": 400}}]', "API key not valid"),
        ("not json at all", "not json at all"),
    ],
)
def test_error_bodies_are_reported_not_raised(body, expected):
    exc = urllib.error.HTTPError(
        "http://example.test/v1/chat/completions",
        400,
        "Bad Request",
        {},  # type: ignore[arg-type]
        io.BytesIO(body.encode()),
    )
    assert expected in _error_detail(exc)


def _completions(handler) -> dict:
    """The body of the one chat request the server received."""
    return next(body for path, body, _ in handler.received if path.endswith("/chat/completions"))


def _script_ok(handler) -> None:
    handler.script["/v1/chat/completions"] = (
        200,
        [json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})],
    )


def test_reasoning_effort_reaches_the_wire_clamped(server):
    """The regression: `effort` was read only by the Anthropic client, so a
    configured level was silently dropped on every OpenAI-compatible call."""
    httpd, handler = server
    _script_ok(handler)
    client = OpenAICompatibleChatClient(
        LLMConfig(provider="openai", model="gemini-3.7-flash", base_url=_base_url(httpd), effort="max")
    )
    client.complete([Message("user", "u")], max_tokens=64)

    # "max" is not in Gemini 3's vocabulary; "high" is the strongest it takes.
    assert _completions(handler)["reasoning_effort"] == "high"


def test_no_configured_effort_leaves_the_field_off_the_request(server):
    httpd, handler = server
    _script_ok(handler)
    client = OpenAICompatibleChatClient(
        LLMConfig(provider="openai", model="gemini-3.7-flash", base_url=_base_url(httpd))
    )
    client.complete([Message("user", "u")], max_tokens=64)

    assert "reasoning_effort" not in _completions(handler)


def test_the_ingest_profile_sends_the_floor_its_model_allows(server):
    httpd, handler = server
    _script_ok(handler)
    config = LLMConfig(provider="openai", model="gemini-3.7-flash", base_url=_base_url(httpd))
    OpenAICompatibleChatClient(config.for_ingest()).complete([Message("user", "u")], max_tokens=64)

    # "off" is what ingest asks for; "low" is as close as Gemini 3 gets.
    assert _completions(handler)["reasoning_effort"] == "low"


def test_anthropic_ignores_the_ingest_off_default():
    """`output_config.effort` has no "off"; omitting it is the old behaviour."""
    pytest.importorskip("anthropic")
    from llmwiki.llm.anthropic_client import AnthropicChatClient

    sent: list[dict] = []

    class _Stream:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def get_final_message(self):
            raise AssertionError("payload capture only")

    client = AnthropicChatClient(LLMConfig(provider="anthropic", model="m", api_key="k", effort="off"))
    client._client.messages.stream = lambda **payload: sent.append(payload) or _Stream()
    with pytest.raises(AssertionError, match="payload capture"):
        client.complete([Message("user", "u")], max_tokens=64)

    assert "output_config" not in sent[0]


# ── a provider that answers, then stalls ──────────────────────────────────
#
# The failure these cover cost a 662-document ingest most of a day. The read
# that stalls happens *after* the response headers arrive, so nothing urllib
# raises is a URLError, and `TimeoutError` went straight past a guard that
# only covered the connect.


class _StallHandler(BaseHTTPRequestHandler):
    """Sends headers promising a body, then never sends the body."""

    received: list = []
    hold = 1.0

    def do_POST(self):
        type(self).received.append(
            json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "4096")
        self.end_headers()
        self.wfile.flush()
        time.sleep(type(self).hold)

    def log_message(self, *args):
        pass


@pytest.fixture
def stalling_server():
    _StallHandler.received = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StallHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, _StallHandler
    httpd.shutdown()


@pytest.fixture
def instant_backoff(monkeypatch):
    monkeypatch.setattr("llmwiki.embeddings.TRANSPORT_BACKOFF", 0.01)


def test_a_stalled_body_is_a_provider_error_not_a_bare_timeout(stalling_server):
    """`response.read()` timing out used to escape as OSError and end the run."""
    httpd, _handler = stalling_server
    url = f"http://127.0.0.1:{httpd.server_port}/v1/embeddings"

    with pytest.raises(ProviderTransportError, match="did not finish the exchange"):
        post_json(url, {"input": ["a"]}, {}, timeout=0.3)

    # Everything that already caught the provider's failures still catches it —
    # in particular the ingest pipeline, which turns it into a per-document
    # warning rather than a crash.
    assert issubclass(ProviderTransportError, ProviderError)


def test_a_stalled_embedding_retries_the_same_batch_and_never_halves_it(
    stalling_server, instant_backoff
):
    """Halving a timeout turns one stalled call into two, then four."""
    httpd, handler = stalling_server
    config = EmbeddingConfig(
        enabled=True,
        model="e",
        base_url=f"http://127.0.0.1:{httpd.server_port}/v1",
        batch_size=8,
        timeout=0.3,
    )

    with pytest.raises(ProviderTransportError):
        embed_texts([f"chunk {i}" for i in range(8)], config)

    assert len(handler.received) == TRANSPORT_ATTEMPTS
    # The batch is re-sent whole every time. A halving cascade would show
    # inputs of 8, 4, 4, 2, 2, ... and 15 requests instead of 3.
    assert [len(body["input"]) for body in handler.received] == [8] * TRANSPORT_ATTEMPTS


def test_a_borrowed_deadline_is_spent_on_the_call_not_on_backoff(
    stalling_server, instant_backoff
):
    """A query lends the embedder what is left of its turn; retries would blow it."""
    httpd, handler = stalling_server
    config = EmbeddingConfig(
        enabled=True, model="e", base_url=f"http://127.0.0.1:{httpd.server_port}/v1"
    )

    with pytest.raises(ProviderTransportError):
        embed_texts(["a"], config, timeout=0.3)

    assert len(handler.received) == 1
