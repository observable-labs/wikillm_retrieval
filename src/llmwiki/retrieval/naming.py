"""How a corpus names its documents — the one wiki-shaped assumption, made explicit.

Retrieval derives four things from a document's *name* rather than from its
text: the key everything else is joined on, the aliases a link or a mention can
reach it by, the string that goes in the 10x-weighted lexical title column, and
whether it is a page at all or a raw source hanging off one.

Every one of those was a statement about markdown files in a wiki directory —
`station-keeping.md` is reachable as `station keeping`, its title column is
`"Station Keeping station keeping"`, and `.md` is stripped from an alias. Eleven
sites held that assumption, none of them said so, and none of them could be
told otherwise.

They are collected here as one strategy object whose default reproduces all
eleven exactly. A corpus that names its documents some other way — opaque ids, a
database primary key, URLs — passes its own and keeps the ranking ladder.

⛔ The default is not "a sensible neutral"; it is *today's behaviour*, verbatim.
A change to any function here changes ranking for every existing caller, which
is the reason they are functions and not constants.

    naming = DocumentNaming(title_field=lambda document: document.title)

is the whole of what a non-wiki corpus usually needs: appending the path stem to
a 10x-weighted column is right for `station-keeping.md` and actively harmful for
an id like `3f9a1c04_quarterly-report`, where it injects the uuid into the field
that dominates BM25.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .keyword import Document

__all__ = [
    "DEFAULT_NAMING",
    "DocumentNaming",
    "default_aliases",
    "default_is_page",
    "default_key",
    "default_surface_forms",
    "default_title_field",
]


def default_key(document: Document) -> str:
    """The identity every lane joins on: the project-relative path."""
    return document.path


def default_aliases(document: Document) -> set[str]:
    """Full path, wiki-relative path, stem, and title.

    The wiki-relative form is what makes `[[concepts/chain-of-thought]]` resolve
    against `wiki/concepts/chain-of-thought.md`.
    """
    wiki_relative = document.path[5:] if document.path.startswith("wiki/") else document.path
    return {
        alias
        for alias in (document.path, wiki_relative, document.stem, document.title)
        if alias
    }


def default_surface_forms(document: Document) -> set[str]:
    """The strings that, appearing in someone else's *prose*, mean this page.

    Deliberately not `aliases`. An alias is what a link may be written as, and
    a link is addressed on purpose; a surface form is what an author types in a
    sentence without meaning to link at all. `wiki/station-keeping.md` is a
    legitimate link target and would be absurd to scan prose for, so the two
    sets are different and always were - `graph.py` and `entities.py` each
    derived their own.
    """
    return {document.title, document.stem, document.stem.replace("-", " ")}


def default_title_field(document: Document) -> str:
    """Title plus the stem read as words, for the lexical title column.

    A hyphenated filename is a second spelling of the title on a wiki, and a
    query naming either should reach the page.
    """
    return f"{document.title} {document.stem.replace('-', ' ')}"


def default_is_page(document: Document) -> bool:
    """Whether this document is a page in its own right rather than raw evidence.

    Only pages contribute aliases and entity surface forms: a raw PDF has no
    name anyone would link to. A corpus whose documents are all first-class
    passes `lambda document: True`.
    """
    return document.kind == "wiki"


@dataclass(frozen=True)
class DocumentNaming:
    """The naming conventions of one corpus, as four functions.

    Frozen and comparable so that "which naming built this index" is a value a
    caller can assert on, not a closure it has to trust.
    """

    key: Callable[[Document], str] = field(default=default_key)
    aliases: Callable[[Document], set[str]] = field(default=default_aliases)
    surface_forms: Callable[[Document], set[str]] = field(default=default_surface_forms)
    title_field: Callable[[Document], str] = field(default=default_title_field)
    is_page: Callable[[Document], bool] = field(default=default_is_page)


#: Today's behaviour, verbatim. Every entry point defaults to it.
DEFAULT_NAMING = DocumentNaming()
