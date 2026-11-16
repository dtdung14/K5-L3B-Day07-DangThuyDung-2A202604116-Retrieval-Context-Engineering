Split: dev · top_k=3 · context budget=800 chars (top-3 context)

| Method | Hit@1 | Evidence Hit@3 | MRR | Context Evidence | Latency ms | Missed (top-3) |
|---|---:|---:|---:|---:|---:|---|
| dense | 15/20 | 17/20 | 0.8 | 12/20 | 55.3 | q06, q11, q14 |
| dense+filter | 16/20 | 18/20 | 0.85 | 12/20 | 52.1 | q11, q14 |
| bm25 | 17/20 | 19/20 | 0.9 | 12/20 | 0.7 | q01 |
| bm25+filter | 18/20 | 19/20 | 0.925 | 12/20 | 0.8 | q01 |
| hybrid | 15/20 | 20/20 | 0.867 | 13/20 | 49.8 | - |
| hybrid+filter | 16/20 | 20/20 | 0.9 | 13/20 | 51.5 | - |
