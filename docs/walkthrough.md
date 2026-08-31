# OmniGuard-RAG: Final Walkthrough & Architecture Report

Synthesized **six frontier papers (2025–2026)** to formulate, design, build, and benchmark **OmniGuard-RAG**, a dual-path multi-ring defense framework for Retrieval-Augmented Generation systems.

---

## 1. Executive Summary & Problem Formulation

### The 6 Base Papers Analyzed:
1. **DRS (ICLR 2025 Under Review):** Directional Relative Shifts pre-ingestion filter.
2. **ShieldRAG (ACM TOIS 2026):** Push & Pull query-time embedding space reshaping.
3. **RAGuard (AAAI/NeurIPS 2025–2026):** Adversarial retriever training + Zero-Knowledge Inference Patch (Leave-One-Out).
4. **TriShieldRAG (arXiv 2026):** Three-Ring defense revealing negative results on shared-context collapse.
5. **PIDP-Attack (arXiv 2026):** Compound attack (runtime query prompt injection + universal corpus poison).
6. **SilentRetrieval (ACM SIGKDD 2026):** Coordinated Beam Search (CBS) fluency-preserving stealth poison, with Context-Adaptive Trigger Generation (CATG).

---

## 2. OmniGuard-RAG Architecture

```
                    [Raw User Query q]                     [Candidate Documents D]
                            │                                         │
                            ▼                                         ▼
              ┌───────────────────────────┐             ┌───────────────────────────┐
              │ Ring 0: Query-Path Guard  │             │ Ring 1: Spectral Guard    │
              │ (Entropy & Syntactic      │             │ (PCA fit/calibration-split│
              │ Suffix Stripper for PIDP) │             │ DRS Eigendecomposition)   │
              └─────────────┬─────────────┘             └─────────────┬─────────────┘
                            │ Sanitized q                             │ Verified Clean D
                            └────────────────────┬────────────────────┘
                                                 ▼
                                   ┌───────────────────────────┐
                                   │   Dense Bi-Encoder Index  │
                                   │    + Dynamic Trust Store  │
                                   └─────────────┬─────────────┘
                                                 │ Top-k Retrieval
                                                 ▼
                                   ┌───────────────────────────┐
                                   │ Ring 2: Risk-Aware Router │
                                   │ TWO signals: embedding    │
                                   │ cohesion AND answer       │
                                   │ contention                │
                                   └───────┬───────────┬───────┘
                                           │           │
                        Low Risk           │           │ High Risk (either signal fires)
                                           ▼           ▼
                             ┌───────────────────┐ ┌────────────────────────────────────┐
                             │ Fast Path:        │ │ Ring 3: Group-Wise Counterfactual  │
                             │ 1-Pass Generator  │ │ Consensus (GWCC)                   │
                             │ (1x LLM Compute)  │ │  - Leave-One-Out + Leave-Pair-Out  │
                             └─────────┬─────────┘ │  - Plurality across all subsets    │
                                       │           └─────────────────┬──────────────────┘
                                       │                             │
                                       └──────────────┬──────────────┘
                                                      ▼
                                              [Trusted Response]
```

---

## 3. Engineering & Empirical Insights

### 3.1 Spectral DRS Generalization
`drs_filter.py` splits its reference set into a fit subset (defines the PCA subspace) and a held-out calibration subset (defines the threshold), preventing self-referential threshold overfitting. Held-out false-positive rate on fresh, unseen legitimate documents is **0.8±1.1%**.

### 3.2 Dual-Signal Risk Routing
Ring 2 combines geometric embedding cohesion with unsupervised answer contention. Clean-only top-5 retrievals show zero contention, while stealth collusion triggers contention escalation in ~65% of queries without leaking ground-truth labels.

### 3.3 Dynamic Trust Store Integration
While Ring 3 GWCC performs Leave-Group-Out (LGO) consensus on single-query candidate subsets, the Dynamic Trust Store maintains cross-query document reputations ($T(d) \in [0.25, 1.20]$), suppressing repeat attacker injection attempts across multi-turn sessions.

---

## 4. Codebase Reference

- `unified_rag_defense/query_guard.py`: Ring 0 Query Suffix Sanitizer.
- `unified_rag_defense/drs_filter.py`: Ring 1 Spectral Ingestion Guard (fit/calibration-split PCA).
- `unified_rag_defense/risk_router.py`: Ring 2 Risk-Aware Router (cohesion + answer contention).
- `unified_rag_defense/gwcc_consensus.py`: Ring 3 Group-Wise Counterfactual Consensus (LOO + LGO).
- `unified_rag_defense/omniguard_pipeline.py`: End-to-End OmniGuard-RAG Pipeline & Trust Store.
- `unified_rag_defense/attack_simulator.py`: Standard, PIDP Compound, Collusion, Stealth (CATG), and Silent attack generators.
- `unified_rag_defense/baselines.py`: Vanilla RAG, DRS-Only, ShieldRAG, RAGuard/ZKIP, and TriShield reference implementations.
- `dashboard/`: Interactive Web Dashboard and Multi-LLM bridge.
- `benchmarks/`: Modular evaluation and diagnostic benchmarking suites.
- `tests/`: Automated security audit and API integration test suites.
