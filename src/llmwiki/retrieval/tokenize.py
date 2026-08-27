"""Query tokenization, ported from llm_wiki's `search.rs::tokenize_query`.

Two behaviours matter and are easy to lose:

* CJK text has no spaces, so a Chinese query is additionally expanded into
  character bigrams plus single characters — otherwise a 4-character query
  matches nothing unless the page repeats it verbatim.
* Stop words are removed from *queries only*, never from documents.
"""

from __future__ import annotations

import re

STOP_WORDS = frozenset(
    {
        "的", "是", "了", "什么", "在", "有", "和", "与", "对", "从",
        "the", "is", "a", "an", "what", "how", "are", "was", "were",
        "do", "does", "did", "be", "been", "being", "have", "has", "had",
        "it", "its", "in", "on", "at", "to", "for", "of", "with", "by",
        "this", "that", "these", "those",
    }
)

# ASCII punctuation + the CJK punctuation the Rust implementation lists.
_SEPARATORS = re.compile(
    r"[\s!-/:-@\[-`{-~，。！？、；：“”‘’（）·～…]+"
)
_CJK_RANGE = re.compile(r"[㐀-鿿]")


def tokenize_query(query: str) -> list[str]:
    """Lowercase tokens, CJK-expanded, stop-words removed, deduplicated."""
    raw = [
        token
        for token in _SEPARATORS.split(query.lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]

    tokens: list[str] = []
    for token in raw:
        characters = list(token)
        if _CJK_RANGE.search(token) and len(characters) > 2:
            tokens.extend(
                characters[i] + characters[i + 1] for i in range(len(characters) - 1)
            )
            tokens.extend(ch for ch in characters if ch not in STOP_WORDS)
            tokens.append(token)
        else:
            tokens.append(token)

    return sorted(set(tokens))


def trim_query_punctuation(value: str) -> str:
    """Strip separator characters from both ends of a query.

    The result is the "query phrase" that phrase-match scoring looks for, so
    `What is RAG?` matches a page containing `what is rag` even though the
    question mark is not in the page.
    """
    return _SEPARATORS.sub(" ", value).strip() if _SEPARATORS.fullmatch(value) else _strip_separators(value)


def _strip_separators(value: str) -> str:
    start, end = 0, len(value)
    while start < end and _SEPARATORS.fullmatch(value[start]):
        start += 1
    while end > start and _SEPARATORS.fullmatch(value[end - 1]):
        end -= 1
    return value[start:end]
