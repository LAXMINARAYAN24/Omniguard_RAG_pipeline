"""
run_omniguard_benchmark.py — 6-system x 7-regime empirical comparison, on a
real text corpus with TF-IDF embeddings and literature attack regimes.

Evaluates:
1. Vanilla RAG (Undefended baseline)
2. DRS Only (ICLR 2025 SVD Spectral Filter)
3. ShieldRAG Only (ACM TOIS 2026 Iterative Reweighting)
4. RAGuard / ZKIP (AAAI 2026 Singleton Leave-One-Out)
5. TriShieldRAG (arXiv 2026 3-Ring Baseline)
6. OmniGuard-RAG (4-Ring Defense with Dynamic Trust Store)
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unified_rag_defense.drs_filter import DRSFilter
from unified_rag_defense.omniguard_pipeline import run_omniguard
from unified_rag_defense import baselines
from unified_rag_defense.bench_common import (
    build_world, run_system, measure_holdout_fpr, DOCS_PER_TOPIC,
)

N_QUERIES = 200
SEED = 7


def dispatch(name, query, pool, world, drs, centroid, trust_store, rng):
    if name == "Vanilla RAG":
        r = baselines.vanilla_rag(query, pool, world)
    elif name == "DRS Only (2025)":
        r = baselines.drs_only(query, pool, drs, world)
    elif name == "ShieldRAG Only (2026)":
        r = baselines.shieldrag_only(query, pool, world)
    elif name == "RAGuard / ZKIP (2026)":
        r = baselines.raguard_zkip(query, pool, world)
    elif name == "TriShield (2026)":
        r = baselines.trishield(query, pool, world, centroid)
    elif name == "OmniGuard-RAG (Ours)":
        r = run_omniguard(query, pool, drs, trust_store, world, rng)
    else:
        raise ValueError(name)
    return r.answer, r.calls


def main():
    world, drs, centroid = build_world(seed=SEED, docs_per_topic=DOCS_PER_TOPIC)
    queries = world.sample_queries(N_QUERIES)
    holdout_fpr = measure_holdout_fpr(world, drs)

    systems = [
        "Vanilla RAG", "DRS Only (2025)", "ShieldRAG Only (2026)",
        "RAGuard / ZKIP (2026)", "TriShield (2026)", "OmniGuard-RAG (Ours)",
    ]

    results = {}
    for name in systems:
        results[name] = run_system(name, world, queries, drs, centroid, dispatch, seed=SEED)

    print(f"Ring 1 (DRS) held-out false-positive rate on fresh, non-malicious docs: {holdout_fpr:.1%}")
    print(f"(n_ref={len(world.clean_docs)} clean docs, dim={world.dim} TF-IDF features, "
          f"n_queries={N_QUERIES})")
    header = (f"{'Defense Framework':<24}| {'Accuracy':>9} | {'Overall ASR':>12} | {'PIDP ASR':>9} | "
              f"{'Collusion ASR':>14} | {'Stealth ASR':>12} | {'Silent ASR':>10} | {'Avg Calls':>9}")
    print("=" * len(header))
    print("EMPIRICAL BENCHMARK EVALUATION (6 SYSTEMS x 6 ATTACK REGIMES, REAL TEXT/TF-IDF)".center(len(header)))
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name in systems:
        t = results[name]
        print(f"{name:<24}| {t.accuracy:>8.1f}% | {t.overall_asr:>11.1f}% | "
              f"{t.regime_asr('pidp'):>8.1f}% | {t.regime_asr('collusion'):>13.1f}% | "
              f"{t.regime_asr('collusion_stealth'):>11.1f}% | "
              f"{t.regime_asr('silent'):>9.1f}% | {t.avg_calls:>9.2f}")
    print("=" * len(header))


if __name__ == "__main__":
    main()
