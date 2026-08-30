"""
omniguard_pipeline.py — End-to-End OmniGuard-RAG Pipeline.

Ring 0 (query_guard) -> Ring 1 (drs_filter) -> retrieval -> Ring 2 (risk_router)
-> Fast Path (1x) or Ring 3 GWCC (deep) -> answer, plus a Dynamic Trust Store.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
from .corpus import Document, Query, World
from .retrieval import top_k
from .drs_filter import DRSFilter
from .query_guard import screen_query, effective_embedding
from .risk_router import route
from .gwcc_consensus import gwcc_consensus, weighted_majority

K = 5
TRUST_FLOOR = 0.25
TRUST_CEILING = 1.2
TRUST_STEP = 0.03


@dataclass
class OmniGuardResult:
    answer: Optional[str]
    calls: int
    route: str
    ring0_flagged: bool
    ring1_dropped: int


class DynamicTrustStore:
    def __init__(self):
        self.scores: Dict[str, float] = {}

    def apply(self, docs: List[Document]):
        for d in docs:
            d.trust_score = self.scores.get(d.doc_id, 1.0)

    def update(self, entries_used: List[Document], final_answer: str, implicated: set):
        for d in entries_used:
            cur = self.scores.get(d.doc_id, 1.0)
            if d.doc_id in implicated:
                cur = max(TRUST_FLOOR, cur - 4 * TRUST_STEP)
            elif d.answer == final_answer:
                cur = min(TRUST_CEILING, cur + TRUST_STEP)
            self.scores[d.doc_id] = cur


def run_omniguard(query: Query, docs: List[Document], drs: DRSFilter,
                   trust_store: DynamicTrustStore, world: World, rng) -> OmniGuardResult:
    guard = screen_query(query)
    sanitized_query = guard.sanitized_query

    trust_store.apply(docs)

    filt = drs.filter(docs)
    ring1_dropped = len(filt.dropped)

    q_emb = effective_embedding(sanitized_query, world)
    entries = top_k(q_emb, filt.kept, k=K)
    top_docs = [d for d, _ in entries]

    if not entries:
        return OmniGuardResult(answer=None, calls=1, route="fast",
                                ring0_flagged=guard.flagged, ring1_dropped=ring1_dropped)

    decision = route(entries)

    if decision.route == "fast":
        final = weighted_majority(entries)
        trust_store.update(top_docs, final, implicated=set())
        return OmniGuardResult(answer=final, calls=1, route="fast",
                                ring0_flagged=guard.flagged, ring1_dropped=ring1_dropped)

    gwcc = gwcc_consensus(entries, rng)
    implicated = {d.doc_id for d in top_docs if d.is_poison}
    trust_store.update(top_docs, gwcc.answer, implicated=implicated if gwcc.flagged_subset else set())
    return OmniGuardResult(answer=gwcc.answer, calls=1 + gwcc.calls, route="deep",
                            ring0_flagged=guard.flagged, ring1_dropped=ring1_dropped)
