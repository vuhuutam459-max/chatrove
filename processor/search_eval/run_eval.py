"""Search-quality benchmark for the Chatrove vault.

Builds a throwaway vault from the synthetic corpus (corpus.json, plus simulated
Smart Librarian output in librarian.json), serves it with the real viewer
(viewer/serve.py) on an ephemeral port, runs every query in queries.json through
both search paths the UI uses, and scores the conversations each one returns:

  search  GET  /api/search?q=...   (global search box, Ctrl+K)
  ask     POST /api/ask            (“Ask the archive” retrieval; the LLM is
                                    disabled so only the retrieved sources count)

Metrics are per query, at conversation level, then averaged:
  hit@1    the top conversation is a grade-2 (primary) match
  R@5      share of relevant conversations in the top 5
  MRR      1 / rank of the first relevant conversation (0 if none in top 10)
  nDCG@10  graded ranking quality
  zero     no results at all;  error  the endpoint failed (HTTP 500)

Offline and deterministic. Run from the repo root:
  python processor/search_eval/run_eval.py                  # table
  python processor/search_eval/run_eval.py --json out.json  # also save per-query results
  python processor/search_eval/run_eval.py --compare processor/search_eval/results.json
  python processor/search_eval/run_eval.py --verbose        # list failing queries
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "processor"))

from processor import parse_and_index as P  # noqa: E402
from processor import librarian as Lib  # noqa: E402
import llm_gateway  # noqa: E402
import serve  # noqa: E402

PATHS = ("search", "ask")
METRICS = ("hit@1", "R@5", "MRR", "nDCG@10", "zero", "error")


def build_vault(tmp: Path) -> Path:
    db = tmp / "eval_vault.db"
    P.CANVAS_DIR = tmp / "canvas"
    conn = P.init_db(db)
    P.process_export_file(conn, HERE / "corpus.json")
    lib = json.loads((HERE / "librarian.json").read_text(encoding="utf-8"))
    for conv_id, meta in lib.items():
        if conv_id.startswith("_"):
            continue
        conn.execute("UPDATE conversations SET summary = ? WHERE id = ?",
                     (meta.get("summary"), conv_id))
        Lib.upsert_tags(conn, conv_id, meta.get("tags", []))
    conn.commit()
    conn.close()
    return db


def _get(base: str, path: str, params: dict):
    url = f"{base}{path}?{urllib.parse.urlencode(params)}"
    try:
        return json.loads(urllib.request.urlopen(url).read())
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"}


def _post(base: str, path: str, payload: dict):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode("utf-8"),
                                 method="POST", headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"}


def _dedupe(ids):
    seen, out = set(), []
    for i in ids:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def run_search(base: str, q: str):
    data = _get(base, "/api/search", {"q": q, "limit": 50})
    if "_error" in data or "error" in data:
        return None, data.get("_error") or data.get("error")
    return _dedupe(r.get("conversation_id") for r in data.get("results", [])), None


def run_ask(base: str, q: str):
    data = _post(base, "/api/ask", {"question": q})
    if "_error" in data:
        return None, data["_error"]
    return _dedupe(s.get("id") for s in data.get("sources", [])), None


def score(ranked, relevant: dict, error) -> dict:
    if ranked is None:
        return {"hit@1": 0.0, "R@5": 0.0, "MRR": 0.0, "nDCG@10": 0.0, "zero": 1.0, "error": 1.0}
    top10 = ranked[:10]
    hit1 = 1.0 if ranked and relevant.get(ranked[0]) == 2 else 0.0
    rel_ids = [c for c, g in relevant.items() if g > 0]
    r5 = sum(1 for c in ranked[:5] if c in relevant) / min(len(rel_ids), 5)
    mrr = next((1.0 / (i + 1) for i, c in enumerate(top10) if c in relevant), 0.0)
    dcg = sum((2 ** relevant.get(c, 0) - 1) / math.log2(i + 2) for i, c in enumerate(top10))
    ideal = sorted(relevant.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return {"hit@1": hit1, "R@5": r5, "MRR": mrr, "nDCG@10": dcg / idcg if idcg else 0.0,
            "zero": 1.0 if not ranked else 0.0, "error": 0.0}


def _mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def summarize(results: list) -> dict:
    """{path: {category|'ALL': {metric: mean}}}"""
    out = {}
    for path in PATHS:
        groups = {"ALL": [r["scores"][path] for r in results]}
        for r in results:
            groups.setdefault(r["category"], []).append(r["scores"][path])
        out[path] = {g: {m: _mean(rows, m) for m in METRICS} | {"n": len(rows)}
                     for g, rows in groups.items()}
    return out


def print_table(summary: dict, compare: dict | None = None):
    for path in PATHS:
        print(f"\n== {path} ==")
        print(f"{'category':<12} {'n':>3} " + " ".join(f"{m:>15}" if compare else f"{m:>8}" for m in METRICS))
        for g in ["ALL"] + sorted(k for k in summary[path] if k != "ALL"):
            row = summary[path][g]
            cells = []
            for m in METRICS:
                if compare and g in compare.get(path, {}):
                    cells.append(f"{compare[path][g][m]:>6.2f} → {row[m]:<6.2f}")
                else:
                    cells.append(f"{row[m]:>8.2f}")
            print(f"{g:<12} {row['n']:>3} " + " ".join(f"{c:>15}" if compare else c for c in cells))


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write per-query results + summary to this file")
    ap.add_argument("--compare", help="results JSON from an earlier run to diff against")
    ap.add_argument("--verbose", action="store_true", help="list queries that miss")
    args = ap.parse_args(argv)

    queries = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))["queries"]
    with tempfile.TemporaryDirectory() as tmpd:
        db = build_vault(Path(tmpd))
        serve.DB_PATH = db
        serve.build_gateway = lambda *a, **k: llm_gateway.NullGateway()
        srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.VaultHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        results = []
        try:
            for q in queries:
                row = {"id": q["id"], "category": q["category"], "query": q["query"],
                       "relevant": q["relevant"], "ranked": {}, "errors": {}, "scores": {}}
                for path, fn in (("search", run_search), ("ask", run_ask)):
                    ranked, err = fn(base, q["query"])
                    row["ranked"][path] = (ranked or [])[:10]
                    if err:
                        row["errors"][path] = err
                    row["scores"][path] = score(ranked, q["relevant"], err)
                results.append(row)
        finally:
            srv.shutdown()
            srv.server_close()

    summary = summarize(results)
    compare = None
    if args.compare:
        compare = json.loads(Path(args.compare).read_text(encoding="utf-8"))["summary"]
    print(f"{len(queries)} queries over {HERE.name}/corpus.json")
    print_table(summary, compare)

    if args.verbose:
        for path in PATHS:
            print(f"\n-- {path}: queries without a primary hit@1 --")
            for r in results:
                s = r["scores"][path]
                if s["hit@1"] < 1:
                    why = r["errors"].get(path) or (", ".join(r["ranked"][path][:3]) or "no results")
                    print(f"  {r['id']:<22} {r['query']!r:<50} -> {why}")

    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "queries": results},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
