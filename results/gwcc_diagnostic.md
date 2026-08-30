# GWCC vs. plain voting -- divergence diagnostic

Tests whether GWCC's consensus verdict, once Ring 2 escalates a query to Ring 3, ever differs from what plain weighted-majority voting on the SAME retrieved top-5 would have given. Run at seed=7, n_queries=200, collusion_stealth attack, k_poison in [3, 5, 8, 12] (3 = benchmark default, the rest deliberately harder).

| k_poison | top-5 poison-count distribution | escalated | GWCC != plain vote | attack success |
|---|---|---|---|---|
| 3 | {0: 77, 1: 101, 2: 18, 3: 4} | 123/200 | 11/123 | 11/200 |
| 5 | {0: 38, 1: 90, 2: 53, 3: 18, 4: 1} | 162/200 | 50/162 | 41/200 |
| 8 | {0: 22, 1: 73, 2: 62, 3: 39, 4: 4} | 178/200 | 75/178 | 56/200 |
| 12 | {0: 9, 1: 39, 2: 71, 3: 61, 4: 20} | 191/200 | 92/191 | 95/200 |

**Total across all tested levels: 228/654 escalations where GWCC's verdict differed from plain voting on the same input.**

See this script's module docstring (`run_gwcc_diagnostic.py`) for the likely mechanism and what this does and does not imply about Ring 2's contention signal (which is independently verified as a correct detector) vs. Ring 3's aggregation rule specifically.