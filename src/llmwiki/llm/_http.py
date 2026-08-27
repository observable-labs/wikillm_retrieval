"""A very small JSON/SSE HTTP helper over the standard library.

The OpenAI-compatible lane deliberately avoids a third-party HTTP dependency:
the whole surface is one POST with either a JSON response or a `data:` event
stream, and keeping it stdlib means `pip install llmwiki` pulls nothing when
you point the tool at a local Ollama.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator

from ..errors import ProviderError

USER_AGENT = "llmwiki/0.1"


def _request(url: str, payload: dict, headers: dict[str, str], timeout: float):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("User-Agent", USER_AGENT)
    for key, value in headers.items():
        if value:
            request.add_header(key, value)
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = _error_detail(exc)
        raise ProviderError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"could not reach {url}: {exc.reason}") from exc


def post_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    with _request(url, payload, headers, timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"{url} returned a non-JSON response: {raw[:200]}") from exc


def post_sse(url: str, payload: dict, headers: dict[str, str], timeout: float) -> Iterator[dict]:
    """Yield parsed `data:` frames until the stream ends or sends [DONE]."""
    with _request(url, payload, headers, timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                # Malformed frames happen with some proxies; skipping one frame
                # loses a token, aborting loses the whole response.
                continue


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - body already consumed
        return exc.reason or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:300]
    # Not every endpoint returns a JSON object here. Gemini's OpenAI-compatible
    # surface answers an unauthenticated request with a top-level *list*, and
    # calling .get on it turned a plain "no API key" into an AttributeError
    # traceback — the error path crashing instead of reporting the error.
    if not isinstance(parsed, dict):
        return str(parsed)[:300]
    error = parsed.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error or parsed)[:300]
