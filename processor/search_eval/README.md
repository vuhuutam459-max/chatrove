# Search quality benchmark

A repeatable, offline benchmark for the vault's two search paths, run against
the real viewer API on a throwaway database:

- `GET /api/search`: the global search box
- `POST /api/ask`: retrieval for "Ask the archive" (the LLM is replaced by
  `NullGateway`, so only the retrieved sources are scored)

All data is synthetic: `corpus.json` (48 made-up conversations in canonical
export format), `librarian.json` (simulated Smart Librarian summaries and tags)
and `queries.json` (81 queries with graded relevance: 2 = the chat the user is
looking for, 1 = also useful). Never add real chat archives here.

```bash
python processor/search_eval/run_eval.py                       # print metrics
python processor/search_eval/run_eval.py --verbose             # + every query without a correct top hit
python processor/search_eval/run_eval.py --compare processor/search_eval/baseline.json
python processor/search_eval/run_eval.py --json out.json       # save a run
```

Metrics are per conversation, per category and overall: hit@1, recall@5, MRR,
nDCG@10, the share of queries returning nothing, and the share that error.

`baseline.json` is the search before the first round of fixes; `results.json`
is the latest saved run. When you change ranking, rerun with `--compare`
against both, and add new queries rather than editing old ones so numbers stay
comparable. The `vi` category covers Vietnamese typed with or without
diacritics, including `đ` vs `d`. The `hard` category was written after the fixes and was not tuned
against; it shows what lexical search still misses (synonyms, cross-lingual).
