"""Query tokenization, ported from llm_wiki's `search.rs::tokenize_query`.

Three behaviours matter and are easy to lose:

* CJK text has no spaces, so a Chinese query is additionally expanded into
  character bigrams plus single characters — otherwise a 4-character query
  matches nothing unless the page repeats it verbatim.
* Stop words are removed from *queries only*, never from documents.
* **The token count is capped.** Query text is the one unbounded input on the
  retrieval hot path, and every token becomes a term in an FTS5 `OR` chain that
  is then matched against every document. Measured: a 20,000-word query is 8 ms
  to tokenize and **200 ms of FTS5 work**, linear in the word count, against a
  measured maximum of **9** tokens across the 44 real questions in the atlas
  suite. Nothing else in the ladder was unbounded — `top_k` is clamped to
  `MAX_TOP_K` and lane depth to `MIN_CANDIDATES` — so this was it.
* **Single-character tokens are dropped, except digits.** The original rule
  dropped every one-character token as noise, which silently discarded the digit
  in `Aurora-1` — separators split it into `aurora` and `1` — and made every
  numbered member of a series identical to the retriever. On a corpus of
  `Aurora-1`, `Aurora-2`, `Borealis-2`, `Borealis-3` that is not an edge case,
  it is most of the corpus, and it was worth 0.12 MRR on the questions that
  name one.
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

# How many distinct terms one query may contribute to a MATCH expression.
#
# ⭐ Not a tuning parameter: the whole point is that it is far above any real
# query and far below a hostile one. Real questions measure at 9 tokens
# (maximum, atlas suite, n=44); this is seven times that, and it is the
# difference between 200 ms and under one for a pasted document.
#
# ⚠️ A refuted hypothesis, recorded because it is the one that looks true: the
# CJK bigram expansion is *not* an amplifier. It multiplies characters, but the
# result is deduplicated, so a 4,000-character repetition of the same phrase
# collapses to 9 tokens like any other query. The exposure was ordinary English
# word count all along.
MAX_QUERY_TOKENS = 64


def contains_cjk(text: str) -> bool:
    """Whether a string holds CJK characters, which segment differently.

    Read by the lexical lane: FTS5's `unicode61` tokenizer treats an unbroken
    CJK run as a single token, so those queries need the substring scorer and
    the bigram expansion below rather than a BM25 index.
    """
    return bool(_CJK_RANGE.search(text))


def tokenize_query(query: str, max_tokens: int = MAX_QUERY_TOKENS) -> list[str]:
    """Lowercase tokens, CJK-expanded, stop-words removed, deduplicated, capped.

    `max_tokens` bounds the work one query may ask of the lexical lane; pass 0
    to lift it. When it binds, the tokens kept are the first distinct ones in
    query order rather than the alphabetically smallest — the front of a query
    is what the asker wrote first, and truncating a sorted set would keep
    whatever happens to start with "a".

    Deduplication is order-preserving so that the cap has something meaningful
    to truncate; for any query under the cap the result is identical to the
    sorted set it has always returned.
    """
    raw = [
        token
        for token in _SEPARATORS.split(query.lower())
        if (len(token) > 1 or token.isdigit()) and token not in STOP_WORDS
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

    distinct = list(dict.fromkeys(tokens))
    if max_tokens and len(distinct) > max_tokens:
        distinct = distinct[:max_tokens]
    return sorted(distinct)


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
