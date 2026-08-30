"""
bench_common.py — Shared benchmark plumbing (Path A).

Before Path A, run_omniguard_benchmark.py owned world-building, the
regime/scenario logic, and the per-system query loop directly. Adding a
second benchmark entry point (the ring ablation ladder) by copy-pasting that
logic would create two copies that could silently drift apart -- e.g. one
gets a bugfix or a regime added and the other doesn't, and nobody notices
because the two scripts never run side by side. This module is the fix:
world construction, REGIMES/build_scenario, and the timed per-system query
loop live in exactly one place, and every benchmark script (the original
single-seed script, the ablation ladder, and the multi-seed evaluation)
imports from here.

The only behavioral change from the pre-Path-A version: SEED is now a
parameter threaded through build_world()/run_system() instead of a
module-level constant closed over by run_system(). That's what makes
multi-seed runs possible without either (a) duplicating run_system per
seed or (b) mutating a global between runs. Also, DRSFilter's own internal
fit/calibration-split seed is now tied to the outer seed (previously
hardcoded to 0 regardless of the benchmark seed) -- otherwise a
"multi-seed" run would still calibrate Ring 1 identically every time,
which isn't the independent-seeds test it claims to be.

Everything else -- REGIMES, REGIME_SALT, build_scenario, fresh_docs, the
deterministic per-(seed, query, regime) RNG derivation -- is copied
verbatim from the validated run_omniguard_benchmark.py so results before
and after this refactor match exactly (checked in verify_refactor.py).
"""
from __future__ import annotations
import copy
import time
import numpy as np
from dataclasses import replace
from typing import Callable, List, Tuple

from .corpus import World, Document, Query
from .attack_simulator import run_attack
from .drs_filter import DRSFilter
from .metrics import Tally
from .omniguard_pipeline import DynamicTrustStore
from . import baselines

REGIMES = ["clean", "standard", "pidp", "collusion_minor", "collusion_major", "collusion_stealth", "silent"]
REGIME_SALT = {name: i for i, name in enumerate(REGIMES)}  # deterministic, unlike Python's hash()

DOCS_PER_TOPIC = 30  # see drs_filter.py / walkthrough.md S3.1 for why this can't be much smaller


def fresh_docs(world: World) -> List[Document]:
    """A per-query working copy of the clean corpus with trust_score reset
    to 1.0 (so one system's trust adjustments from an earlier query can't
    leak into another query or another system's run).

    PERFORMANCE (Path A): originally copy.deepcopy(world.clean_docs), which
    recursively copies every field of all 480 Document objects INCLUDING
    each one's TF-IDF embedding array -- measured at ~26s of a ~30s single-
    system run (89%), because it's called once per (query, regime), i.e.
    1400 times per system per seed. trust_score is the only field any code
    in this package ever mutates on a document after corpus construction
    (verified by grep -- corpus.py sets .embedding exactly once, at
    ingestion; nothing else assigns to .embedding, .doc_id, .text, .label,
    .answer, .is_poison, or .attack_type anywhere). dataclasses.replace()
    below creates a new Document per call (so trust_score is still
    independent per copy, preserving the no-leakage property) but shares
    every other field, including the embedding array, by reference instead
    of copying it -- safe because nothing mutates it, and ~40x faster in
    practice. Verified byte-identical benchmark output before/after (see
    verify_refactor.py)."""
    return [replace(d, trust_score=1.0) for d in world.clean_docs]


def build_scenario(world: World, query: Query, regime: str, rng):
    if regime == "clean":
        return query, [], False
    if regime == "collusion_minor":
        q2, docs = run_attack("collusion", world, query, rng, k_poison=2)
        return q2, docs, True
    if regime == "collusion_major":
        q2, docs = run_attack("collusion", world, query, rng, k_poison=3)
        return q2, docs, True
    if regime == "collusion_stealth":
        q2, docs = run_attack("collusion_stealth", world, query, rng, k_poison=3)
        return q2, docs, True
    q2, docs = run_attack(regime, world, query, rng)
    return q2, docs, True


def bucket_for_regime(regime: str) -> str:
    """collusion_minor and collusion_major are reported together as
    'collusion'; every other regime reports under its own name."""
    if regime in ("collusion_minor", "collusion_major"):
        return "collusion"
    return regime


def build_world(seed: int, docs_per_topic: int = DOCS_PER_TOPIC, drs_seed: int | None = None,
                 embedding_space: str = "tfidf"):
    """One World + one DRSFilter + the TriShield centroid.

    drs_seed controls DRSFilter's own fit/calibration-split seed
    (drs_filter.py's CALIBRATION_FRACTION shuffle), independent of the
    corpus/query seed. Defaults to 0, matching every result already
    documented in walkthrough.md -- so the single-seed sanity script
    (run_omniguard_benchmark.py) keeps reproducing those exact numbers.
    run_full_evaluation.py explicitly passes drs_seed=seed instead: across
    a multi-seed run, tying DRS's own calibration split to the SAME fixed
    seed every time would mean the "independent seeds" stability check
    wasn't actually testing Ring 1's calibration variance at all, only
    query/corpus variance -- a real (if subtle) gap in how independent the
    earlier 3-seed check actually was.

    embedding_space ("tfidf" default, or "lsa"): PATH B -- passed straight
    through to World; see corpus.py's module-level LSA_COMPONENTS comment
    and World's docstring for what "lsa" actually changes and why. Default
    is unchanged so every existing call site (Path A's scripts, none of
    which pass this argument) keeps building TF-IDF worlds exactly as
    before -- verified byte-identical in verify_refactor.py.
    """
    world = World(docs_per_topic=docs_per_topic, seed=seed, embedding_space=embedding_space)
    drs = DRSFilter(reference_docs=world.clean_docs, seed=drs_seed if drs_seed is not None else 0)
    centroid = baselines._corpus_centroid(world)
    return world, drs, centroid


def measure_holdout_fpr(world: World, drs: DRSFilter, n_per_topic: int = 5, seed: int = 999) -> float:
    """Fresh, non-malicious documents the filter has never seen. See
    drs_filter.py's module docstring for why this number has to be reported
    alongside every ASR figure, not just the ASR figures alone."""
    from .text_gen import make_sentence
    rng = np.random.default_rng(seed)
    fresh = []
    for t, topic in enumerate(world.topics):
        for _ in range(n_per_topic):
            text = make_sentence(topic, rng, wrong=False)
            fresh.append(Document(doc_id="fresh", text=text, embedding=world.embed(text),
                                   topic_id=t, label="correct", answer=topic["answer"]))
    return drs.holdout_false_positive_rate(fresh)


AnswerFn = Callable[[str, Query, List[Document], World, DRSFilter, np.ndarray, DynamicTrustStore, np.random.Generator],
                     Tuple[object, int]]


def run_system(name: str, world: World, queries: List[Query], drs: DRSFilter, centroid: np.ndarray,
               answer_fn: AnswerFn, seed: int) -> Tally:
    """Runs one system/variant over all queries x REGIMES for a given seed.

    Timing scope: only the answer_fn call itself (the defense pipeline's own
    processing) is timed, not build_scenario (attack construction is shared
    benchmark-harness overhead every system pays identically, not something
    any defense is being evaluated on). See run_full_evaluation.py for the
    caveat on what this wall-clock number does and doesn't represent.
    """
    tally = Tally()
    trust_store = DynamicTrustStore()  # only OmniGuard-RAG's dispatch branch actually uses this
    for qi, query in enumerate(queries):
        for regime in REGIMES:
            rng = np.random.default_rng(seed * 1000 + qi * 10 + REGIME_SALT[regime])
            eff_query, poison_docs, attacked = build_scenario(world, query, regime, rng)
            pool = fresh_docs(world) + poison_docs
            t0 = time.perf_counter()
            answer, calls = answer_fn(name, eff_query, pool, world, drs, centroid, trust_store, rng)
            elapsed = time.perf_counter() - t0
            bucket = bucket_for_regime(regime)
            tally.record(bucket, attacked, query.correct_answer, answer, calls, elapsed=elapsed)
    return tally
