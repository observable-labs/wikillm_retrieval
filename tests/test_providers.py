"""The OpenAI-compatible adapters, driven against a real local HTTP server.

Stub clients verify the pipeline; these verify the wire — SSE accumulation,
error surfacing, URL derivation, and the embeddings batch contract.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llmwiki.config import EmbeddingConfig, LLMConfig
from llmwiki.embeddings import embed_texts
from llmwiki.errors import ProviderError
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
