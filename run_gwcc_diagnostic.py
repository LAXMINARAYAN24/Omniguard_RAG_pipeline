"""
run_gwcc_diagnostic.py — does GWCC's consensus verdict ever actually differ
from plain weighted-majority voting on the same retrieved set? (Path A)

walkthrough.md S3.5 claims: "GWCC catches something the other rings can't
... it resolves cases the fast path alone would not." That claim has two
parts. Part 1 -- Ring 3 gets invoked specifically because of the
contention signal, on cases cohesion alone would wave through -- IS
verified (see run_ablation_benchmark.py: cohesion-only never escalates on
collusion_stealth; both-signals escalates in exactly the queries where
poison reaches the top-5 window). Part 2 -- that GWCC's OWN consensus
mechanism, once invoked, recovers a different (correct) answer than plain
weighted_majority on that identical retrieved set would have given -- was
asserted but not directly tested at the time.

This script tests part 2 directly: for every query that gets escalated to
Ring 3, compute BOTH the GWCC answer and what plain weighted_majority on
the exact same top-k entries would have given, and count divergences.

RESULT (measured, all seeds): GWCC's verdict never diverges from plain
voting -- not in the standard benchmark (0/21 escalations across all 7
regimes, seed=7, n=200), and not even under a deliberately strengthened
collusion_stealth attack up to k_poison=12 (0 divergences at every tested
level, despite the top-5 window frequently containing a poison PLURALITY
or MAJORITY by that point -- see the poison-count distributions printed
below). Attack success climbs steadily with k_poison (poison eventually
wins outright), but GWCC never outperforms the plain vote on the way there.

WHY, most likely: GWCC's final answer is a plurality vote ACROSS 12
sub-evaluations (1 full-set + 5 leave-one-out + 6 sampled leave-pair-out,
at k=5), each of which is ITSELF a weighted_majority vote. If poison holds
enough combined similarity*trust weight to win the full-set vote, removing
just 1-2 of the (typically 5) retrieved documents at a time often still
leaves poison holding a plurality within that sub-evaluation too -- so
poison tends to also win most of the 12 sub-votes, not just the full one.
This is a property of the "plurality-of-sub-majorities" aggregation rule
itself, not of TF-IDF vs. real embeddings -- it depends on vote
combinatorics, not vector representation, so it is not a "check on real
embeddings" question (contrast with Path B's other open questions).

WHAT THIS DOES AND DOES NOT MEAN: it does not mean Ring 2's contention
signal is wrong (that detection is independently verified above), and it
does not mean the pipeline's 0% stealth ASR is fake (it's real and
reproducible -- see run_full_evaluation.py). It means that 0% is currently
earned by (a) DRS/Ring1 catching non-stealth collusion before Ring 2/3 ever
engage, and (b) the DynamicTrustStore changing which documents reach the
top-5 window in later queries -- NOT by GWCC's consensus step recovering
an answer the fast path would have missed on the same input. That's a
narrower, more precisely-supported claim than the one currently written
down, and the gap is worth closing (either in the write-up's wording, or
by changing GWCC's aggregation rule) before this goes in the final report.

Usage:
    python run_gwcc_diagnostic.py
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np

from unified_rag_defense.bench_common import build_world, fresh_docs
from unified_rag_defense.attack_simulator import apply_collusion_stealth
from unified_rag_defense.query_guard import screen_query, effective_embedding
from unified_rag_defense.retrieval import top_k
from unified_rag_defense.gwcc_consensus import weighted_majority, gwcc_consensus
from unified_rag_defense.risk_router import route
from unified_rag_defense.metrics import WRONG_ANSWER_TAG

SEED = 7
N_QUERIES = 200
K_POISON_LEVELS = [3, 5, 8, 12]  # 3 = the benchmark's own default; higher = deliberately harder


def main():
    world, drs, centroid = build_world(seed=SEED)
    queries = world.sample_queries(N_QUERIES)

    print(f"{'k_poison':>8} | {'top-5 poison-count distribution':<38} | {'escalated':>9} | "
          f"{'GWCC != plain vote':>19} | {'attack success':>14}")
    print("-" * 100)

    rows = []
    for k_poison in K_POISON_LEVELS:
        escalated = 0
        divergences = 0
        poison_counts = Counter()
        successes = 0

        for qi, query in enumerate(queries):
            rng = np.random.default_rng(SEED * 1000 + qi * 10 + 999 + k_poison)
            eff_query, poison_docs = apply_collusion_stealth(world, query, rng, k_poison=k_poison)
            pool = fresh_docs(world) + poison_docs

            filt = drs.filter(pool)
            q_emb = effective_embedding(screen_query(eff_query).sanitized_query, world)
            entries = top_k(q_emb, filt.kept, k=5)
            if not entries:
                continue
            top_docs = [d for d, _ in entries]
            poison_counts[sum(1 for d in top_docs if d.is_poison)] += 1

            decision = route(entries)
            plain = weighted_majority(entries)
            if decision.route == "deep":
                escalated += 1
                rng_g = np.random.default_rng(qi + 777)
                gwcc = gwcc_consensus(entries, rng_g)
                if gwcc.answer != plain:
                    divergences += 1
                final = gwcc.answer
            else:
                final = plain
            if final == WRONG_ANSWER_TAG:
                successes += 1

        dist = dict(sorted(poison_counts.items()))
        print(f"{k_poison:>8} | {str(dist):<38} | {escalated:>6}/200 | {divergences:>16}/{escalated:<3}"
              f" | {successes:>9}/200")
        rows.append((k_poison, dist, escalated, divergences, successes))

    print()
    print("If 'GWCC != plain vote' is 0 at every k_poison level above, Ring 3's consensus")
    print("step is not currently changing any outcome vs. a plain fast-path vote on the same")
    print("retrieved set -- see this script's module docstring for what that does and does")
    print("not imply, and for the likely mechanism.")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    total_escalated = sum(r[2] for r in rows)
    total_div = sum(r[3] for r in rows)
    lines = [
        "# GWCC vs. plain voting -- divergence diagnostic",
        "",
        f"Tests whether GWCC's consensus verdict, once Ring 2 escalates a query to Ring 3, "
        f"ever differs from what plain weighted-majority voting on the SAME retrieved top-5 "
        f"would have given. Run at seed={SEED}, n_queries={N_QUERIES}, "
        f"collusion_stealth attack, k_poison in {K_POISON_LEVELS} (3 = benchmark default, "
        f"the rest deliberately harder).",
        "",
        "| k_poison | top-5 poison-count distribution | escalated | GWCC != plain vote | attack success |",
        "|---|---|---|---|---|",
    ]
    for k_poison, dist, escalated, divergences, successes in rows:
        lines.append(f"| {k_poison} | {dist} | {escalated}/200 | {divergences}/{escalated} | {successes}/200 |")
    lines.append("")
    lines.append(f"**Total across all tested levels: {total_div}/{total_escalated} escalations where GWCC's "
                 f"verdict differed from plain voting on the same input.**")
    lines.append("")
    lines.append("See this script's module docstring (`run_gwcc_diagnostic.py`) for the likely mechanism "
                 "and what this does and does not imply about Ring 2's contention signal (which is "
                 "independently verified as a correct detector) vs. Ring 3's aggregation rule specifically.")
    (results_dir / "gwcc_diagnostic.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {results_dir / 'gwcc_diagnostic.md'}")


if __name__ == "__main__":
    main()
