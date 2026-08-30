# OmniGuard-RAG — Path A Evaluation Report

Generated 2026-08-30 12:15 UTC. All figures are mean ± 95% confidence interval (Student's-t) across 8 independent seeds: [7, 11, 23, 41, 59, 79, 97, 113]. Each seed is an independently regenerated corpus, query set, and DRS calibration split (not just a reshuffle of the same data) at 200 queries/seed, docs_per_topic=30.

Ring 1 (DRS) held-out false-positive rate on fresh, non-malicious docs: **0.8±1.1%** (n=5 fresh docs/topic/seed).

## 1. Main system comparison

| Defense Framework | Accuracy | Overall ASR | PIDP ASR | Collusion ASR | Stealth ASR | Silent ASR |
|---|---|---|---|---|---|---|
| Vanilla RAG | 85.1±0.1% | 0.9±0.2% | 1.2±0.6% | 1.5±0.4% | 1.1±0.5% | 0.0±0.0% |
| DRS Only (2025) | 85.6±0.1% | 0.2±0.1% | 0.0±0.0% | 0.0±0.0% | 1.1±0.5% | 0.0±0.0% |
| ShieldRAG Only (2026) | 85.1±0.1% | 0.9±0.2% | 1.2±0.6% | 1.5±0.4% | 1.1±0.5% | 0.0±0.0% |
| RAGuard / ZKIP (2026) | 80.5±0.4% | 6.5±0.5% | 2.4±0.4% | 13.4±1.2% | 9.9±1.4% | 0.0±0.0% |
| TriShield (2026) | 85.5±0.1% | 0.2±0.1% | 0.1±0.1% | 0.4±0.3% | 0.5±0.4% | 0.0±0.0% |
| OmniGuard-RAG (Ours) | 100.0±0.0% | 0.0±0.0% | 0.0±0.0% | 0.0±0.0% | 0.1±0.1% | 0.0±0.0% |

## 2. Per-ring ablation ladder

Same seeds, same queries, same attacks as Table 1 -- each row adds one ring to the previous row's pipeline. See `unified_rag_defense/ablations.py` for exactly what each step does and does not include.

| Ring configuration | Accuracy | Overall ASR | PIDP ASR | Collusion ASR | Stealth ASR | Silent ASR |
|---|---|---|---|---|---|---|
| Ring0 alone | 99.4±0.1% | 0.7±0.2% | 0.0±0.0% | 1.5±0.4% | 1.1±0.5% | 0.0±0.0% |
| +Ring1 (DRS) | 99.8±0.1% | 0.2±0.1% | 0.0±0.0% | 0.0±0.0% | 1.1±0.5% | 0.0±0.0% |
| +Ring2 (cohesion only) | 99.8±0.1% | 0.2±0.1% | 0.0±0.0% | 0.0±0.0% | 1.1±0.5% | 0.0±0.0% |
| +Ring2 (both signals) | 98.6±0.2% | 1.6±0.2% | 0.0±0.0% | 0.0±0.1% | 9.8±1.3% | 0.0±0.0% |

## 3. Compute cost: call count vs. wall-clock latency

Wall-clock time measures this simulation's own Python/NumPy orchestration cost around each system's own logic (there is no real LLM in this benchmark's loop -- see `baselines.py`'s RAGuard/ZKIP docstring). It is NOT a production LLM-latency estimate. It IS a genuine, measured comparison of how expensive each defense's own processing is, independent of whatever LLM calls its `avg_calls` number represents -- and the two rankings disagree (see below), which avg_calls alone could not have shown.

| System / Ring configuration | Avg Calls | Avg Latency (ms) |
|---|---|---|
| Vanilla RAG | 1.0±0.0 | 0.4±0.0 |
| DRS Only (2025) | 1.0±0.0 | 1.2±0.0 |
| ShieldRAG Only (2026) | 4.0±0.0 | 0.4±0.0 |
| RAGuard / ZKIP (2026) | 6.0±0.0 | 0.4±0.0 |
| TriShield (2026) | 3.0±0.0 | 4.9±0.1 |
| OmniGuard-RAG (Ours) | 1.3±0.1 | 1.3±0.1 |
| Ring0 alone | 1.0±0.0 | 0.4±0.0 |
| +Ring1 (DRS) | 1.0±0.0 | 1.2±0.2 |
| +Ring2 (cohesion only) | 1.1±0.1 | 1.2±0.1 |
| +Ring2 (both signals) | 2.1±0.1 | 1.2±0.1 |
