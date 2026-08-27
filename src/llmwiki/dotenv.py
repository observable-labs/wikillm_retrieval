"""Load `.env` files into the process environment.

The core package is dependency-free, so this is a small `set -a; . ./.env`
in Python rather than a `python-dotenv` import. It exists because the
alternative failure is silent and expensive: with no `LLMWIKI_*` variables
set, `config._detect_provider()` falls through to Anthropic and the run dies
on a credential the user never intended to use, while their key sits unread
in a file two directories up.

Search order, first definition winning:

    $LLMWIKI_DOTENV                        (explicit path; skips the rest)
    <project>/.env
    $PWD/.env, then each parent below $HOME
    ~/.config/llmwiki/.env                 (user-global)

An already-set environment variable is never overwritten, so `.env` is a
*floor*, not a ceiling: `LLMWIKI_MODEL=x llmwiki ask ...` still wins, and so
does anything exported by the surrounding shell. Within the resolution chain
in `config.py` the values land as environment variables, which means they
outrank both config files — the same position they'd occupy if the user had
sourced the file by hand.

`LLMWIKI_DOTENV=0` (or `false`/`no`/`off`) disables loading entirely.

Supported syntax is the intersection of what shells and dotenv libraries
agree on: `KEY=value`, an optional `export ` prefix, `#` comments, single
quotes (literal), double quotes (with `\\n`-style escapes), and `$VAR` /
`${VAR}` expansion against earlier lines and the real environment. Multi-line
values and command substitution are not supported — a line that doesn't parse
is reported on stderr rather than dropped in silence.

`LLMWIKI_STRICT_PROJECT` is deliberately out of scope: it decides *which*
project to open, so it cannot be read from a file found by way of the project.
Set it in the real environment or the user config.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DotenvResult", "load_dotenv", "parse_dotenv", "dotenv_candidates"]

DISABLED = frozenset({"0", "false", "no", "off"})

# `export FOO=bar`, `FOO = bar`, `FOO=`. The name is restricted to what a
# shell would accept so a stray prose line doesn't parse as an assignment.
ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0"}


@dataclass
class DotenvResult:
    """What a load actually did — no values, only names."""

    paths: list[Path] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def dotenv_candidates(project_dir: Path | None = None) -> list[Path]:
    """Files to try, in precedence order. Does not check existence."""
    explicit = os.environ.get("LLMWIKI_DOTENV", "").strip()
    if explicit and explicit.lower() not in DISABLED:
        return [Path(explicit).expanduser()]

    candidates: list[Path] = []
    if project_dir is not None:
        candidates.append(Path(project_dir) / ".env")

    # Walk up from $PWD, but stop below $HOME: a user-global file belongs in
    # the config directory, not in a `~/.env` that probably belongs to
    # something else.
    home = Path.home()
    try:
        current = Path.cwd().resolve()
    except OSError:  # cwd deleted out from under us
        current = None
    while current is not None and current != current.parent and current != home:
        candidates.append(current / ".env")
        current = current.parent

    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else home / ".config"
    candidates.append(root / "llmwiki" / ".env")

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def parse_dotenv(text: str, lookup=None) -> tuple[dict[str, str], list[str]]:
    """Parse `.env` text into values plus warnings for lines that didn't parse.

    `lookup(name)` resolves `$VAR` references that the file itself hasn't
    defined yet; it defaults to the real environment.
    """
    if lookup is None:
        lookup = os.environ.get

    values: dict[str, str] = {}
    warnings: list[str] = []

    def resolve(name: str) -> str:
        if name in values:
            return values[name]
        return lookup(name) or ""

    for number, raw in enumerate(text.replace("\r\n", "\n").split("\n"), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = ASSIGNMENT.match(raw)
        if match is None:
            warnings.append(f"line {number}: not a KEY=value assignment; ignored")
            continue
        key, rest = match.group(1), match.group(2)
        value, error = _read_value(rest, resolve)
        if error is not None:
            warnings.append(f"line {number} ({key}): {error}; ignored")
            continue
        values[key] = value

    return values, warnings


def load_dotenv(project_dir: Path | None = None, *, stream=None) -> DotenvResult:
    """Apply the first `.env` found for each name, without overwriting the env."""
    result = DotenvResult()
    if os.environ.get("LLMWIKI_DOTENV", "").strip().lower() in DISABLED:
        return result

    for path in dotenv_candidates(project_dir):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        result.paths.append(path)
        values, warnings = parse_dotenv(text)
        result.warnings.extend(f"{path}: {warning}" for warning in warnings)

        for key, value in values.items():
            # Both guards matter: a real environment variable outranks the
            # file, and an earlier (higher-precedence) file outranks a later
            # one — `applied` is what makes the second half true.
            if os.environ.get(key) not in (None, "") or key in result.applied:
                result.skipped.append(key)
                continue
            os.environ[key] = value
            result.applied.append(key)

    if result.warnings:
        out = sys.stderr if stream is None else stream
        for warning in result.warnings:
            print(f"llmwiki: {warning}", file=out)

    return result


def _read_value(rest: str, resolve) -> tuple[str, str | None]:
    """Read one right-hand side. Returns (value, error)."""
    text = rest.lstrip()
    if not text:
        return "", None

    quote = text[0] if text[0] in "\"'" else None
    if quote is None:
        return _expand(_strip_comment(text.rstrip()), resolve), None

    body, closed, tail = _split_quoted(text[1:], quote)
    if not closed:
        return "", f"unterminated {quote} quote (multi-line values are not supported)"
    if _strip_comment(tail.strip()).strip():
        return "", "trailing text after the closing quote"

    # Single quotes are literal all the way down, exactly as in a shell.
    return (body if quote == "'" else _expand(body, resolve, escapes=True)), None


def _split_quoted(text: str, quote: str) -> tuple[str, bool, str]:
    body: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        # A backslash only escapes inside double quotes; in single quotes it
        # is an ordinary character, so `'a\'` ends at the quote.
        if char == "\\" and quote == '"' and index + 1 < len(text):
            body.append(char)
            body.append(text[index + 1])
            index += 2
            continue
        if char == quote:
            return "".join(body), True, text[index + 1 :]
        body.append(char)
        index += 1
    return "".join(body), False, ""


def _strip_comment(text: str) -> str:
    """Drop a trailing `#` comment, which must be preceded by whitespace."""
    for index, char in enumerate(text):
        if char == "#" and (index == 0 or text[index - 1] in " \t"):
            return text[:index].rstrip()
    return text


def _expand(text: str, resolve, *, escapes: bool = False) -> str:
    """Substitute `$VAR` / `${VAR}`, honouring `\\$` and (optionally) `\\n`."""
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            nxt = text[index + 1]
            if nxt == "$" or (escapes and nxt in "\\\"'"):
                out.append(nxt)
                index += 2
                continue
            if escapes and nxt in ESCAPES:
                out.append(ESCAPES[nxt])
                index += 2
                continue
            out.append(char)
            index += 1
            continue
        if char == "$" and index + 1 < len(text):
            if text[index + 1] == "{":
                end = text.find("}", index + 2)
                name = text[index + 2 : end] if end != -1 else ""
                if end != -1 and NAME.fullmatch(name):
                    out.append(resolve(name))
                    index = end + 1
                    continue
            else:
                match = NAME.match(text, index + 1)
                if match is not None:
                    out.append(resolve(match.group(0)))
                    index = match.end()
                    continue
        out.append(char)
        index += 1
    return "".join(out)
