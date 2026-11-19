"""Ranked, crash-proof search over the vault (shared by the viewer's
global search, "Ask the archive" retrieval and the conversations filter).

User input never reaches FTS5 MATCH verbatim: it is split into word tokens,
English stopwords are dropped, and each token is quoted, so punctuation such
as ``C++``, ``node.js``, ``can't`` or ``ssh:`` can no longer raise an FTS5
syntax error. Power users can still type FTS5 syntax (``"exact phrase"``,
``AND``/``OR``/``NOT``, ``NEAR``, ``term*``); it is used as-is when valid.

Ranking is per conversation, not per message: for every query term we take the
conversation's best-matching message (bm25), add the term's bm25 against the
conversation's title + Librarian summary, plus a bonus when a Librarian tag
matches, and sum over terms. A chat that covers more of the query therefore
wins even when the words are spread over several messages.
"""

from __future__ import annotations

import re
import sqlite3

STOPWORDS = frozenset("""
a about above after again all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from
further had has have having he her here hers him his how i if in into is it its
itself just me more most my myself no nor not of off on once only or other our ours
out over own same she should so some such than that the their theirs them then there
these they this those through to too under until up very was we were what when
where which while who whom why will with would you your yours yourself
s t d ll m re ve
""".split())

TITLE_WEIGHT = 2.0      # bm25 column weight for conversations_fts.title
SUMMARY_WEIGHT = 1.0    # bm25 column weight for conversations_fts.summary
TAG_BONUS = 3.0         # added per query term that matches a Librarian tag
MSG_CANDIDATES = 500    # per-term cap on matching messages considered

_ADVANCED = re.compile(r'"|\*\s*$|\b(AND|OR|NOT|NEAR)\b')
_WORD = re.compile(r"\w+", re.UNICODE)
_COMPOUND = re.compile(r"\w+(?:[/.\-']\w+)*", re.UNICODE)


def query_terms(text: str) -> list[str]:
    """Lower-cased, de-duplicated word tokens with stopwords removed
    (all tokens are kept if the query is nothing but stopwords)."""
    words = []
    for chunk in _COMPOUND.findall(text or ""):
        parts = _WORD.findall(chunk)
        # "N/A" -> the phrase "n a": its letters alone would match everywhere.
        if len(parts) > 1 and all(len(p) == 1 for p in parts):
            words.append(" ".join(parts).lower())
        else:
            words.extend(p.lower() for p in parts)
    words = [w.replace("_", " ").strip() for w in words]
    words = [w for w in words if w]
    kept = [w for w in words if w not in STOPWORDS] or words
    # A lone ASCII letter ("c" from "C++", "s" from "it's") matches nearly
    # everything, so it only counts when the query has nothing longer.
    kept = [w for w in kept if len(w) > 1 or not w.isascii()] or kept
    return list(dict.fromkeys(kept))


def _term_expr(term: str, prefix: bool) -> str:
    return f'"{term}"*' if prefix else f'"{term}"'


def _wants_prefix(term: str, is_last: bool) -> bool:
    # Last token: the user may still be typing (search-as-you-type).
    # Non-ASCII tokens (Russian, Vietnamese, ...) get no Porter stemming, so a
    # prefix match stands in for it on longer words ("договор" ~ "договора").
    if is_last and len(term) >= 2:
        return True
    return len(term) >= 4 and not term.isascii()


def term_exprs(terms: list[str]) -> list[str]:
    return [_term_expr(t, _wants_prefix(t, i == len(terms) - 1)) for i, t in enumerate(terms)]


def safe_match(text: str) -> str:
    """An OR-of-terms FTS5 expression that is always valid ('' if no terms)."""
    return " OR ".join(term_exprs(query_terms(text)))


def is_advanced(text: str) -> bool:
    return bool(_ADVANCED.search(text or ""))


def _valid_match(conn: sqlite3.Connection, table: str, expr: str) -> bool:
    try:
        conn.execute(f"SELECT 1 FROM {table} WHERE {table} MATCH ? LIMIT 1", (expr,)).fetchall()
        return True
    except sqlite3.OperationalError:
        return False


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone() is not None


def rank_conversations(conn: sqlite3.Connection, text: str, limit: int = 50) -> list[dict]:
    """Conversations ranked for `text`: [{'id', 'score', 'match'}], best first.
    `match` is the FTS5 expression to use for snippets of this query."""
    terms = query_terms(text)
    if not terms:
        return []
    exprs = term_exprs(terms)
    scores: dict[str, float] = {}

    def add(cid, s):
        scores[cid] = scores.get(cid, 0.0) + s

    advanced = is_advanced(text) and _valid_match(conn, "messages_fts", text)
    if advanced:
        # Explicit FTS5 syntax: honour it as one clause.
        clauses = [text]
        match = text
    else:
        clauses = exprs
        match = " OR ".join(exprs)

    for clause in clauses:
        rows = conn.execute("""
            SELECT m.conversation_id AS cid, MIN(s) AS best FROM (
                SELECT rowid AS rid, bm25(messages_fts) AS s FROM messages_fts
                WHERE messages_fts MATCH ? ORDER BY s LIMIT ?
            ) hits JOIN messages m ON m.id = hits.rid
            GROUP BY m.conversation_id
        """, (clause, MSG_CANDIDATES)).fetchall()
        for cid, best in rows:
            add(cid, -best)

    if _has_table(conn, "conversations_fts"):
        meta_clauses = clauses if _valid_match(conn, "conversations_fts", clauses[0]) else exprs
        for clause in meta_clauses:
            rows = conn.execute(f"""
                SELECT c.id, bm25(conversations_fts, {TITLE_WEIGHT}, {SUMMARY_WEIGHT})
                FROM conversations_fts JOIN conversations c ON c.rowid = conversations_fts.rowid
                WHERE conversations_fts MATCH ?
            """, (clause,)).fetchall()
            for cid, s in rows:
                add(cid, -s)

    tag_rows = conn.execute(
        "SELECT ct.conversation_id, t.name FROM conversation_tags ct "
        "JOIN tags t ON t.id = ct.tag_id").fetchall()
    for cid, name in tag_rows:
        parts = _WORD.findall(name.lower().replace("_", " "))
        hits = sum(1 for t in terms if any(p == t or (len(t) >= 3 and p.startswith(t)) for p in parts))
        # With explicit syntax a tag alone admits a chat only when it covers every
        # term and nothing is excluded ("kubernetes" yes, "python NOT java" no).
        tag_satisfies = hits == len(terms) and not re.search(r"\bNOT\b", text)
        if hits and (not advanced or cid in scores or tag_satisfies):
            add(cid, TAG_BONUS * hits)

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [{"id": cid, "score": s, "match": match} for cid, s in ranked]


def best_message(conn: sqlite3.Connection, conv_id: str, match: str,
                 open_tag: str = "<mark>", close_tag: str = "</mark>", tokens: int = 40):
    """The conversation's best-matching message for `match` with a snippet, or None."""
    try:
        return conn.execute("""
            SELECT m.conversation_id, m.role, m.content, m.seq,
                   snippet(messages_fts, 0, ?, ?, '…', ?) AS snippet
            FROM messages_fts JOIN messages m ON messages_fts.rowid = m.id
            WHERE messages_fts MATCH ? AND m.conversation_id = ?
            ORDER BY rank LIMIT 1
        """, (open_tag, close_tag, tokens, match, conv_id)).fetchone()
    except sqlite3.OperationalError:
        return None


def canvas_match(conn: sqlite3.Connection, text: str) -> str:
    """FTS5 expression for canvas_fts: the user's syntax if valid, else safe OR."""
    if is_advanced(text) and _valid_match(conn, "canvas_fts", text):
        return text
    return safe_match(text)
