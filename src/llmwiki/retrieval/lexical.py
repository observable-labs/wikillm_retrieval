"""L2: a lexical lane that ranks by term *importance*, not term presence.

The scorer this replaces (`keyword._token_match_score`) counted how many query
tokens appeared anywhere in a document, by substring. No frequency, no inverse
document frequency, no length normalization, and `art` matched `cartesian`. On a
corpus where one term is in every document and another is in two, it weighted
them identically — which is the whole reason plain BM25 was beating the pipeline
that was supposed to contain it.

SQLite's FTS5 ships `bm25()` in the standard library, so this costs no
dependency and no hand-maintained N/avgdl/df bookkeeping. The structural bonuses
the old scorer added ad hoc (filename +200, phrase-in-title +50, token-in-title
+5) survive as **column weights**, which is where a signal like that belongs: a
title match is worth more per occurrence, not worth a flat constant regardless of
how well the rest of the document matches.

**Source damping does not survive, and that is a measured decision.**
`SOURCE_SCORE_FACTOR = 0.6` existed because the substring scorer had no length
normalization: a short raw source repeating the query terms beat the compiled
page on raw counts, so the page needed protecting. BM25 normalizes by document
length and weights the title column ten times the body, which produces the same
ordering on its own — the wiki-outranks-source case still holds with the factor
removed. Keeping it cost 0.43 MRR on `intra-doc` questions, whose gold *is* a
raw source, for no gain anywhere else. It remains applied on the substring lane,
which still needs it.

**CJK queries** keep that lane: `unicode61` does not segment Chinese, so a CJK
query would match only on whole runs, and `tokenize_query`'s bigram expansion
plus substring matching is genuinely the better retriever there.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .keyword import Document
from .tokenize import contains_cjk

# Title / headings / body. The ratio, not the magnitude, is what matters; these
# are the build-plan's values and they reproduce the old lane's ordering
# intuition (a title hit beats a body hit) without its flat constants.
TITLE_WEIGHT = 10.0
HEADINGS_WEIGHT = 5.0
BODY_WEIGHT = 1.0

SCHEMA = """
CREATE VIRTUAL TABLE lexical USING fts5(
    doc_id UNINDEXED,
    title,
    headings,
    body,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


@dataclass(frozen=True)
class LexicalHit:
    path: str
    score: float


def extract_headings(content: str) -> str:
    """Markdown ATX headings, joined. The middle-weight column.

    A heading is a claim about what a section is *about*, which is exactly the
    signal a body-frequency score under-weights and a title-only score misses.
    """
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            lines.append(stripped.lstrip("#").strip())
    return "\n".join(lines)


def match_expression(tokens: list[str]) -> str:
    """Quote every token and OR them.

    Raw user text must never reach MATCH: FTS5 has an operator syntax, so
    unquoted input is both a crash (`AND` as a bare token) and an injection.
    """
    quoted = ['"' + token.replace('"', '""') + '"' for token in tokens if token.strip()]
    return " OR ".join(quoted)


class LexicalIndex:
    """An in-memory FTS5 index over one set of documents.

    Built from `Document` objects rather than from disk so it composes with the
    existing loader and stays honest under the growth protocol: the index cannot
    disagree with the documents it was built from, because it *is* them.
    """

    def __init__(self, documents: list[Document]):
        self._connection = sqlite3.connect(":memory:")
        self._connection.executescript(SCHEMA)
        self._connection.executemany(
            "INSERT INTO lexical (doc_id, title, headings, body) VALUES (?, ?, ?, ?)",
            [
                (
                    document.path,
                    f"{document.title} {document.stem.replace('-', ' ')}",
                    extract_headings(document.content),
                    document.content,
                )
                for document in documents
            ],
        )
        self._connection.commit()
        self.entries = len(documents)

    def search(self, tokens: list[str], limit: int) -> list[LexicalHit]:
        expression = match_expression(tokens)
        if not expression:
            return []
        try:
            rows = self._connection.execute(
                "SELECT doc_id, bm25(lexical, ?, ?, ?) AS score "
                "FROM lexical WHERE lexical MATCH ? ORDER BY score LIMIT ?",
                (TITLE_WEIGHT, HEADINGS_WEIGHT, BODY_WEIGHT, expression, max(1, limit)),
            ).fetchall()
        except sqlite3.OperationalError:
            # A query that survives quoting but not parsing is a miss, not a
            # crash: retrieval degrading to zero results is recoverable, a
            # traceback out of the hot path is not.
            return []

        # bm25() is negative and smaller-is-better; negate for the usual
        # higher-is-better convention.
        return [LexicalHit(path=doc_id, score=-float(score)) for doc_id, score in rows]

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "LexicalIndex":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def usable_for(query: str) -> bool:
    """Whether FTS5 can serve this query, or the substring lane must.

    `unicode61` treats an unbroken CJK run as one token, so a four-character
    Chinese query matches only a document repeating it verbatim. The old
    substring scorer with `tokenize_query`'s bigram expansion is strictly better
    there, and it is still the right answer until a segmenting tokenizer exists.
    """
    return not contains_cjk(query)


__all__ = [
    "BODY_WEIGHT",
    "HEADINGS_WEIGHT",
    "TITLE_WEIGHT",
    "LexicalHit",
    "LexicalIndex",
    "extract_headings",
    "match_expression",
    "usable_for",
]
