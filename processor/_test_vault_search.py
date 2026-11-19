"""Tests for processor/vault_search.py and the search-index migration.
Run:  python processor/_test_vault_search.py
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "processor"))

from processor import parse_and_index as P  # noqa: E402
import vault_search as VS  # noqa: E402

OLD_FTS = """
CREATE VIRTUAL TABLE messages_fts USING fts5(content, content=messages, content_rowid=id,
    tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE canvas_fts USING fts5(content, content=canvas_artifacts, content_rowid=rowid,
    tokenize='unicode61 remove_diacritics 2');
"""


def _seed(conn):
    conn.execute("INSERT INTO accounts(email, first_seen, last_export) VALUES ('t@example.com','x','x')")
    for cid, title, msgs in [
        ("c-k8s", "Pod keeps restarting", ["My pod is restarting", "Check kubectl logs --previous"]),
        ("c-cpp", "Linked list crash", ["C++ segfault on delete", "Use valgrind"]),
        ("c-bread", "Sourdough troubleshooting", ["Loaf came out dense"]),
    ]:
        conn.execute("INSERT INTO conversations(id, account_id, title) VALUES (?, 1, ?)", (cid, title))
        for seq, text in enumerate(msgs):
            conn.execute("INSERT INTO messages(conversation_id, seq, role, content) VALUES (?,?,?,?)",
                         (cid, seq, "user", text))
    conn.commit()


def test_query_terms():
    assert VS.query_terms("What's the C++ error?") == ["c", "error"]
    assert VS.query_terms("the") == ["the"]
    assert VS.safe_match("ssh: node.js") == '"ssh" OR "node" OR "js"*'
    assert VS.safe_match("договор аренды") == '"договор"* OR "аренды"*'
    assert VS.safe_match("!!!") == ""
    print("  OK  query_terms / safe_match")


def test_ranking_and_safety():
    with tempfile.TemporaryDirectory() as tmp:
        conn = P.init_db(Path(tmp) / "v.db")
        _seed(conn)
        top = lambda q: [h["id"] for h in VS.rank_conversations(conn, q)][:1]
        for q in ['C++', 'node.js', "can't", 'ssh:', '#N/A', '(', '"', 'AND', 'a-b', '*']:
            VS.rank_conversations(conn, q)  # must never raise
        assert top("restarts") == ["c-k8s"], "porter stemming"
        assert top("kube") == ["c-k8s"], "prefix on last term"
        assert top("sourdough") == ["c-bread"], "title is searchable"
        assert top("C++ segfault") == ["c-cpp"], "punctuation"
        assert top('"kubectl logs"') == ["c-k8s"], "advanced FTS syntax still works"
        conn.execute("UPDATE conversations SET summary = 'About fermentation' WHERE id = 'c-bread'")
        assert top("fermentation") == ["c-bread"], "summary index follows updates"
        conn.close()
    print("  OK  ranking, stemming, prefix, title/summary, no FTS errors")


def test_migrates_old_tokenizer():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "old.db"
        conn = P.init_db(db)
        _seed(conn)
        conn.close()
        # Swap in an old-style (pre-porter, no conversations_fts) index, then migrate.
        conn = sqlite3.connect(db)
        conn.executescript("DROP TABLE messages_fts; DROP TABLE conversations_fts;" + OLD_FTS.split(";")[0] + ";")
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")
        conn.commit()
        conn.close()
        conn = P.init_db(db)
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='messages_fts'").fetchone()[0]
        assert "porter" in sql, sql
        assert [h["id"] for h in VS.rank_conversations(conn, "restarts")][:1] == ["c-k8s"]
        assert [h["id"] for h in VS.rank_conversations(conn, "sourdough")][:1] == ["c-bread"]
        conn.close()
    print("  OK  old unicode61 index is rebuilt with porter + conversations_fts backfilled")


if __name__ == "__main__":
    test_query_terms()
    test_ranking_and_safety()
    test_migrates_old_tokenizer()
    print("\nvault_search: ALL OK")
