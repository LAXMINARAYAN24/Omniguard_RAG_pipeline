"""
ablations.py — Path A per-ring ablation ladder for OmniGuard-RAG.

Each step below adds exactly one ring to the previous step's pipeline, so
the gap between consecutive rows in the report is that ring's own
contribution, isolated:

    Ring0 alone            -- Ring 0 (query-suffix guard) + retrieval +
                               plain weighted-majority vote. No DRS
                               filtering, no risk routing, no GWCC.
    +Ring1 (DRS)            -- adds Ring 1 (drs_filter.DRSFilter.filter):
                               poison documents that carry a spectral tell
                               are dropped from the candidate pool BEFORE
                               retrieval, same as the full pipeline.
    +Ring2 (cohesion only)  -- adds Ring 2's risk router, but with ONLY the
                               embedding-cohesion signal active (matches
                               risk_router.cohesion; answer_contention is
                               computed but never allowed to trigger
                               escalation). Escalated queries go to Ring 3
                               (GWCC); everything else takes the fast-path
                               vote, exactly like the full pipeline's
                               "fast" branch.
    +Ring2 (both signals)   -- adds the SECOND signal, answer_contention,
                               so a query escalates if EITHER cohesion OR
                               contention fires. This is risk_router.route()
                               exactly as run_omniguard uses it -- so this
                               step's own numbers are a direct check that
                               the ablation ladder's last row and the full
                               OmniGuard-RAG system are actually built from
                               the same routing logic, not a hand-copied
                               approximation of it.

WHAT'S DELIBERATELY EXCLUDED, AND WHY: none of the four steps touch the
DynamicTrustStore (no .apply(), no .update() -- every step retrieves
against docs at their default trust_score=1.0, refreshed every query by
bench_common.fresh_docs, exactly like every OTHER system/baseline in this
benchmark). The full OmniGuard-RAG pipeline (omniguard_pipeline.run_omniguard)
DOES use a persistent trust store across queries within a run. That's a
real, separate mechanism -- it reweights which documents win retrieval in
LATER queries based on EARLIER queries' outcomes, which is a cross-query
effect, not a per-ring processing step. Mixing it into a per-ring ladder
would conflate "what does adding Ring 2 do" with "what does having seen
N earlier queries do," which isn't a clean per-ring comparison. So: any
gap between this ladder's last row ("+Ring2 (both signals)", which uses
the identical Ring0+Ring1+Ring2+Ring3 logic run_omniguard uses) and the
full "OmniGuard-RAG (Ours)" row in the main system-comparison table is the
DynamicTrustStore's own contribution, specifically, not a discrepancy or
an error in this ladder.

Each step's answer/calls accounting matches run_omniguard's own convention
exactly (1 call for the fast path or the "no candidates survived" case;
1 + gwcc.calls for the deep path), so avg_calls is directly comparable
between the ablation ladder and the main system table.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from .corpus import Document, Query, World
from .drs_filter import DRSFilter
from .retrieval import top_k
from .query_guard import screen_query, effective_embedding
from .risk_router import cohesion, answer_contention, RISK_THRESHOLD, CONTENTION_THRESHOLD
from .gwcc_consensus import weighted_majority, gwcc_consensus

K = 5

VARIANTS = ["Ring0 alone", "+Ring1 (DRS)", "+Ring2 (cohesion only)", "+Ring2 (both signals)"]


@dataclass
class AblationResult:
    answer: Optional[str]
    calls: int


def _fast_or_deep(entries, use_router: bool, use_contention: bool, rng) -> AblationResult:
    """Shared tail for every step that has candidates to vote over: decide
    fast vs. deep (or always fast, if use_router is False -- i.e. Ring 0/1
    alone, which have no router yet), then vote."""
    if not entries:
        return AblationResult(answer=None, calls=1)

    if use_router:
        top_docs = [d for d, _ in entries]
        coh = cohesion(top_docs)
        cont = answer_contention(entries) if use_contention else 0.0
        risky = (coh < RISK_THRESHOLD) or (use_contention and cont >= CONTENTION_THRESHOLD)
    else:
        risky = False

    if not risky:
        return AblationResult(answer=weighted_majority(entries), calls=1)

    gwcc = gwcc_consensus(entries, rng)
    return AblationResult(answer=gwcc.answer, calls=1 + gwcc.calls)


def _ring0_only(query: Query, docs: List[Document], world: World, rng) -> AblationResult:
    guard = screen_query(query)
    q_emb = effective_embedding(guard.sanitized_query, world)
    entries = top_k(q_emb, docs, k=K)
    return _fast_or_deep(entries, use_router=False, use_contention=False, rng=rng)


def _ring0_ring1(query: Query, docs: List[Document], drs: DRSFilter, world: World, rng) -> AblationResult:
    guard = screen_query(query)
    filt = drs.filter(docs)
    q_emb = effective_embedding(guard.sanitized_query, world)
    entries = top_k(q_emb, filt.kept, k=K)
    return _fast_or_deep(entries, use_router=False, use_contention=False, rng=rng)


def _ring2(query: Query, docs: List[Document], world: World, drs: DRSFilter, rng,
           use_contention: bool) -> AblationResult:
    """Shared body for both Ring2 variants -- the only difference between
    '+Ring2 (cohesion only)' and '+Ring2 (both signals)' is whether
    answer_contention is allowed to trigger escalation, so both are one
    function with a flag rather than two near-duplicate copies."""
    guard = screen_query(query)
    filt = drs.filter(docs)
    q_emb = effective_embedding(guard.sanitized_query, world)
    entries = top_k(q_emb, filt.kept, k=K)
    return _fast_or_deep(entries, use_router=True, use_contention=use_contention, rng=rng)


def dispatch_ablation(name: str, query: Query, pool: List[Document], world: World,
                       drs: DRSFilter, centroid, trust_store, rng) -> tuple:
    """Same call signature as run_omniguard_benchmark.py's dispatch() /
    run_full_evaluation.py's dispatch_main() -- (name, query, pool, world,
    drs, centroid, trust_store, rng) -> (answer, calls) -- so
    bench_common.run_system can drive ablation variants and main systems
    identically. `centroid` and `trust_store` are accepted for interface
    parity but unused (see module docstring: no ablation step touches the
    trust store, and none needs TriShield's centroid)."""
    if name == "Ring0 alone":
        r = _ring0_only(query, pool, world, rng)
    elif name == "+Ring1 (DRS)":
        r = _ring0_ring1(query, pool, drs, world, rng)
    elif name == "+Ring2 (cohesion only)":
        r = _ring2(query, pool, world, drs, rng, use_contention=False)
    elif name == "+Ring2 (both signals)":
        r = _ring2(query, pool, world, drs, rng, use_contention=True)
    else:
        raise ValueError(name)
    return r.answer, r.calls
