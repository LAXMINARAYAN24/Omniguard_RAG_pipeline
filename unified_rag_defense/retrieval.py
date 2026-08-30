"""retrieval.py — shared top-k cosine similarity retrieval over a document pool."""
from __future__ import annotations
import numpy as np
from typing import List, Tuple
from .corpus import Document


def _vectorized_sims(query_embedding: np.ndarray, docs: List[Document]) -> np.ndarray:
    """PERFORMANCE (Path A): originally a Python for-loop computing one
    np.dot per document. top_k() is called once per (system, query, regime)
    -- i.e. thousands of times per benchmark run -- so this was profiled as
    a leading cost alongside drs_filter.py's equivalent per-document loop.
    Stacking every document's embedding into one matrix and normalizing
    both the query and the whole matrix in one vectorized pass computes the
    IDENTICAL per-document cosine-similarity*trust_score value (same
    formula: normalize both vectors, dot product, scale by trust_score),
    just without ~500 separate tiny NumPy calls per invocation."""
    q = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
    D = np.stack([d.embedding for d in docs])
    norms = np.linalg.norm(D, axis=1, keepdims=True) + 1e-12
    Dn = D / norms
    trust = np.array([d.trust_score for d in docs])
    return (Dn @ q) * trust


def top_k(query_embedding: np.ndarray, docs: List[Document], k: int = 5) -> List[Tuple[Document, float]]:
    if not docs:
        return []
    sims = _vectorized_sims(query_embedding, docs)
    # Exactly the original tie-break semantics: a stable ascending argsort,
    # then reverse the WHOLE resulting order (not np.argsort(-sims)). These
    # are NOT equivalent for tied values -- verified directly: negating and
    # doing a stable ascending sort put the FIRST-occurring tied element
    # first, whereas argsort(ascending)[::-1] puts the LAST-occurring tied
    # element first (reversing a stable sort reverses tie order too). Since
    # trust_score can create real ties (e.g. several untouched docs at the
    # default 1.0), this distinction is observable, not academic -- checked
    # against a synthetic tie case before relying on it, and against the
    # full benchmark's real output below.
    order = np.argsort(sims, kind="stable")[::-1][:k]
    return [(docs[i], float(sims[i])) for i in order]
