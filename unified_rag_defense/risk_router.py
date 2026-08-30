"""
risk_router.py — Ring 2: Risk-Aware Router.

Two INDEPENDENT unsupervised risk signals, not one:

  1. Embedding cohesion -- mean pairwise cosine similarity among the top-k
     retrieved embeddings. Catches anything that disturbs retrieval
     geometry (off-topic attractor docs, a stylistically different poison
     doc).

  2. Answer contention -- the weighted vote-mass fraction that disagrees
     with the top-k's plurality answer (same doc.answer / similarity*trust
     weighting weighted_majority already uses everywhere else, so this is
     not a new privileged signal -- every system in this benchmark already
     reads doc.answer as the LLM-extractable claim of a passage). Catches
     the thing cohesion structurally cannot: a "true stealth" collusion
     attack (attack_simulator.apply_collusion_stealth) is, by
     construction, textually and geometrically indistinguishable from
     clean topic content -- that's what makes it stealthy. Measured
     directly: clean-only top-5 retrievals in this corpus NEVER show any
     contention (every genuine topic document agrees on the topic's real
     answer, so plurality share is always 100%), while camouflaged
     collusion pushes contention well above 0 in ~2/3 of cases once the
     poison text is query-targeted enough to actually compete for
     retrieval slots. This is the same failure mode TriShieldRAG documents
     as its own negative result (Ring 2/3 "false consensus": agreement
     stayed high while ASR was high) -- a single geometric agreement
     signal is blind to an attack optimized against a *different* axis
     (conclusion, not geometry). Route to Ring 3 if EITHER signal fires.
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple
from .corpus import Document

RISK_THRESHOLD = 0.55     # cohesion floor: measured, clean-only top-5 retrievals never go below ~0.58 (n=30 sample)
CONTENTION_THRESHOLD = 0.15  # measured: clean-only top-5 retrievals show exactly 0.0 contention always (n=200);
                              # any threshold > 0 has zero observed false-route risk in this corpus.


def cohesion(top_docs: List[Document]) -> float:
    if len(top_docs) < 2:
        return 1.0
    embs = np.stack([d.embedding / (np.linalg.norm(d.embedding) + 1e-12) for d in top_docs])
    sim_matrix = embs @ embs.T
    iu = np.triu_indices(len(top_docs), k=1)
    return float(np.mean(sim_matrix[iu]))


def answer_contention(entries: List[Tuple[Document, float]]) -> float:
    """1 - (plurality answer's weighted vote share). 0 when every retrieved
    doc agrees; grows toward 1 as retrieved docs support conflicting
    conclusions."""
    tally = defaultdict(float)
    total = 0.0
    for d, w in entries:
        ww = max(w, 0.0) * d.trust_score
        tally[d.answer] += ww
        total += ww
    if total <= 0 or len(tally) < 2:
        return 0.0
    return 1.0 - (max(tally.values()) / total)


@dataclass
class RouteDecision:
    route: str
    delta: float
    contention: float


def route(entries: List[Tuple[Document, float]]) -> RouteDecision:
    top_docs = [d for d, _ in entries]
    coh = cohesion(top_docs)
    delta = 1.0 - coh
    cont = answer_contention(entries)
    risky = (coh < RISK_THRESHOLD) or (cont >= CONTENTION_THRESHOLD)
    return RouteDecision(route="deep" if risky else "fast", delta=delta, contention=cont)
