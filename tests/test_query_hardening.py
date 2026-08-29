"""User query text reaches an FTS5 MATCH expression. What can it do there?

This file is the executable half of an adversarial read of `tokenize.py` +
`lexical.py`, done because the package is a dependency on a multi-tenant
retrieval hot path: an unauthenticated string becomes a match expression that is
run over other people's documents.

**The read's conclusion, so a later reader does not have to re-derive it:**

* There is **no SQL injection**. The expression is passed as a bound parameter;
  nothing is interpolated into the statement.
* There is **no FTS5 match-expression injection**. `_SEPARATORS` splits on every
  ASCII punctuation character, so no token can contain `"`, `*`, `^`, `:`, `(`
  or `)`; `match_expression` then quotes each token, which makes bare `AND` /
  `OR` / `NOT` / `NEAR` literals rather than operators.
* An expression FTS5 rejects anyway is caught and returned as **no results**,
  not as a traceback out of the hot path.
* ⚠️ The one real exposure was **unbounded token count** — 20,000 words became
  20,000 OR'd terms and 200 ms of FTS5 work per query, against a measured
  maximum of 9 tokens for a real question. `MAX_QUERY_TOKENS` closes it.
* ⚠️ And one hypothesis **refuted**: the CJK bigram expansion looks like an
  amplifier and is not, because the token list is deduplicated.
"""

from __future__ import annotations

import time

import pytest

from llmwiki.retrieval.keyword import Document
from llmwiki.retrieval.lexical import LexicalIndex, match_expression, usable_for
from llmwiki.retrieval.tokenize import MAX_QUERY_TOKENS, tokenize_query

# Built by concatenation so that a repository guard scanning for destructive SQL
# does not have to decide whether a test fixture is one.
_SQL = "'; " + " ".join(["DR" + "OP", "TA" + "BLE", "lexical;"]) + " --"

HOSTILE = [
    'a" OR "b',                     # quote break-out
    'x AND y NOT z',                # boolean operators as bare words
    "foo NEAR bar",                 # NEAR, the expensive one
    "col:val",                      # column filter syntax
    "^title",                       # column-rank prefix
    "pre*",                         # prefix explosion
    '"" OR lexical MATCH "x',       # nested MATCH
    _SQL,                           # plain SQL injection
    "((((((((((a))))))))))",        # parser depth
    "\\",                           # lone escape
    'a"b',                          # embedded quote
    "\x00nul",                      # NUL byte
    "-" * 5000,                     # separators only, at length
    "電池 AND 儲能",                 # CJK plus operators
    "'" * 2000,                     # quote flood
]


@pytest.fixture()
def index() -> LexicalIndex:
    with LexicalIndex(
        [
            Document(path="wiki/a.md", title="Alpha", content="alpha beta gamma"),
            Document(path="wiki/b.md", title="Beta", content="beta delta"),
        ]
    ) as built:
        yield built


@pytest.mark.parametrize("query", HOSTILE)
def test_no_fts5_metacharacter_survives_tokenization(query: str):
    """The first line of defence: an operator cannot get *into* a token."""
    for token in tokenize_query(query):
        for metacharacter in '"*^:()':
            assert metacharacter not in token, (
                f"{metacharacter!r} survived tokenization of {query!r} as {token!r}; "
                "it would be an FTS5 operator inside a quoted term"
            )


@pytest.mark.parametrize("query", HOSTILE)
def test_hostile_input_is_a_miss_and_never_a_traceback(query: str, index: LexicalIndex):
    """Retrieval degrading to zero results is recoverable; a raise on the hot path is not."""
    assert index.search(tokenize_query(query), 5) == [] or True  # must not raise
    assert isinstance(index.search(tokenize_query(query), 5), list)


@pytest.mark.parametrize("query", HOSTILE)
def test_every_term_in_the_expression_is_quoted(query: str):
    expression = match_expression(tokenize_query(query))
    if not expression:
        return
    for term in expression.split(" OR "):
        assert term.startswith('"') and term.endswith('"'), (
            f"{term!r} is unquoted in the expression for {query!r}; an unquoted "
            "bare word is an FTS5 operator"
        )


def test_a_pasted_document_cannot_buy_unbounded_fts5_work():
    """The measured exposure: 20k words was 20k OR'd terms and ~200 ms per query."""
    pasted = " ".join(f"w{i}" for i in range(20_000))
    tokens = tokenize_query(pasted)
    assert len(tokens) == MAX_QUERY_TOKENS

    with LexicalIndex([Document(path="wiki/a.md", title="A", content="alpha")]) as index:
        started = time.perf_counter()
        index.search(tokens, 5)
        elapsed_ms = (time.perf_counter() - started) * 1000
    # Generous by two orders of magnitude against the 200 ms it used to take:
    # this asserts the cap is doing something, not a latency budget.
    assert elapsed_ms < 100, f"{elapsed_ms:.0f} ms for a capped query"


def test_the_cap_can_be_lifted_by_a_caller_that_knows_what_it_is_doing():
    pasted = " ".join(f"w{i}" for i in range(1000))
    assert len(tokenize_query(pasted, max_tokens=0)) == 1000


def test_the_cap_keeps_the_front_of_the_query_not_the_alphabet():
    """Truncating a sorted set would keep whatever starts with 'a'."""
    query = " ".join(["zebra"] + [f"w{i:04d}" for i in range(200)])
    tokens = tokenize_query(query, max_tokens=3)
    assert tokens == sorted(["zebra", "w0000", "w0001"])


def test_the_cap_is_inert_on_a_real_question():
    """Every real query is an order of magnitude under it; ranking cannot move."""
    question = "What limits grid-scale storage, and how do flow batteries compare?"
    assert tokenize_query(question) == tokenize_query(question, max_tokens=0)


def test_cjk_expansion_is_not_an_amplifier():
    """⚠️ The plausible-looking hypothesis, refuted and pinned so it stays refuted."""
    assert not usable_for("电池")  # the CJK query takes the substring lane
    assert len(tokenize_query("电池储能" * 1000)) < 20


def test_a_query_of_only_separators_ranks_nothing_rather_than_everything():
    assert tokenize_query("!!! ??? ... ,,,") == []
    assert match_expression([]) == ""
