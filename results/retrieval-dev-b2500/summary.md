Split: dev · top_k=3 · context budget=2500 chars (top-3 context)

| Method | Hit@1 | Evidence Hit@3 | MRR | Context Evidence | Latency ms | Missed (top-3) |
|---|---:|---:|---:|---:|---:|---|
| dense | 15/20 | 17/20 | 0.8 | 17/20 | 73.6 | q06, q11, q14 |
| dense+filter | 16/20 | 18/20 | 0.85 | 18/20 | 45.7 | q11, q14 |
| bm25 | 17/20 | 19/20 | 0.9 | 19/20 | 0.4 | q01 |
| bm25+filter | 18/20 | 19/20 | 0.925 | 19/20 | 0.4 | q01 |
| hybrid | 15/20 | 20/20 | 0.867 | 20/20 | 54.5 | - |
| hybrid+filter | 16/20 | 20/20 | 0.9 | 20/20 | 87.4 | - |
