"""
run_embedding_comparison.py — Path B: does OmniGuard-RAG's defense hold up
under a genuinely different embedding geometry, not just TF-IDF?

WHAT "REAL EMBEDDINGS" MEANS HERE, AND WHY: this environment has no network
access, so a pretrained neural embedding model (sentence-transformers, an
API-based embedding service, etc.) is not installable. This script compares
TF-IDF (sparse representation) against LSA/TruncatedSVD (dense, 100 components,
fit on the same TF-IDF matrix).

Usage:
    python benchmarks/run_embedding_comparison.py                    # default: 3 seeds x 200 queries, both spaces
    python benchmarks/run_embedding_comparison.py --seeds 7 11 23 41 59 79 97 113  # full Path-A-parity run
    python benchmarks/run_embedding_comparison.py --recalibrate-risk-threshold
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unified_rag_defense.bench_common import (
    build_world, run_system, measure_holdout_fpr, fresh_docs, REGIMES,
)
from unified_rag_defense.ablations import VARIANTS as ABLATION_VARIANTS, dispatch_ablation
from unified_rag_defense.stats_utils import summarize, Stat
from unified_rag_defense.query_guard import screen_query, effective_embedding
from unified_rag_defense.retrieval import top_k
from unified_rag_defense.risk_router import cohesion, answer_contention, RISK_THRESHOLD, CONTENTION_THRESHOLD
from benchmarks.run_full_evaluation import MAIN_SYSTEMS, METRICS, dispatch_main, tally_to_metrics, aggregate, _row

DEFAULT_SEEDS = [7, 11, 23]
DEFAULT_N_QUERIES = 200
SPACES = ["tfidf", "lsa"]


def measure_clean_statistics(world, n_queries: int = 200, seed: int = 7) -> Dict[str, float]:
    """Reproduces the exact measurement risk_router.py's own comments cite
    for RISK_THRESHOLD/CONTENTION_THRESHOLD (clean-only top-5 retrievals'
    cohesion/contention distribution), but against WHATEVER world is
    passed in -- so it can be run identically against both embedding
    spaces rather than trusting that a TF-IDF-derived number still
    applies."""
    import numpy as np
    queries = world.sample_queries(n_queries)
    docs = fresh_docs(world)
    cohesions, contentions = [], []
    for query in queries:
        guard = screen_query(query)
        q_emb = effective_embedding(guard.sanitized_query, world)
        entries = top_k(q_emb, docs, k=5)
        top_docs = [d for d, _ in entries]
        cohesions.append(cohesion(top_docs))
        contentions.append(answer_contention(entries))
    cohesions = np.array(cohesions)
    contentions = np.array(contentions)
    return {
        "cohesion_min": float(cohesions.min()), "cohesion_mean": float(cohesions.mean()),
        "cohesion_max": float(cohesions.max()),
        "contention_min": float(contentions.min()), "contention_mean": float(contentions.mean()),
        "contention_max": float(contentions.max()),
    }


def run_one_space(embedding_space: str, seeds: List[int], n_queries: int, docs_per_topic: int) -> dict:
    raw_main = {name: defaultdict(list) for name in MAIN_SYSTEMS}
    raw_ablation = {name: defaultdict(list) for name in ABLATION_VARIANTS}
    holdout_fprs = []
    clean_stats_per_seed = []

    for seed in seeds:
        world, drs, centroid = build_world(seed=seed, docs_per_topic=docs_per_topic, drs_seed=seed,
                                            embedding_space=embedding_space)
        queries = world.sample_queries(n_queries)
        holdout_fprs.append(100.0 * measure_holdout_fpr(world, drs))
        clean_stats_per_seed.append(measure_clean_statistics(world, n_queries=n_queries, seed=seed))

        for name in MAIN_SYSTEMS:
            t = run_system(name, world, queries, drs, centroid, dispatch_main, seed=seed)
            for k, v in tally_to_metrics(t).items():
                raw_main[name][k].append(v)
        for name in ABLATION_VARIANTS:
            t = run_system(name, world, queries, drs, centroid, dispatch_ablation, seed=seed)
            for k, v in tally_to_metrics(t).items():
                raw_ablation[name][k].append(v)

    return {
        "embedding_space": embedding_space, "seeds": seeds,
        "main_raw": {k: dict(v) for k, v in raw_main.items()},
        "ablation_raw": {k: dict(v) for k, v in raw_ablation.items()},
        "holdout_fpr_per_seed": holdout_fprs,
        "clean_stats_per_seed": clean_stats_per_seed,
    }


def print_side_by_side(title: str, names: List[str], agg_a: dict, agg_b: dict, label_a: str, label_b: str):
    header = (f"{'System / Ring configuration':<24}| {'Acc(' + label_a + ')':>11} | {'Acc(' + label_b + ')':>11} | "
              f"{'ASR(' + label_a + ')':>11} | {'ASR(' + label_b + ')':>11}")
    print("=" * len(header))
    print(title.center(len(header)))
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name in names:
        sa, sb = agg_a[name], agg_b[name]
        print(f"{name:<24}| {str(sa['accuracy']):>11} | {str(sb['accuracy']):>11} | "
              f"{str(sa['overall_asr']):>11} | {str(sb['overall_asr']):>11}")
    print("=" * len(header))


def write_markdown_report(path: Path, seeds, n_queries, docs_per_topic,
                           results_by_space: Dict[str, dict], agg_main_by_space, agg_abl_by_space,
                           holdout_stat_by_space, clean_stat_by_space):
    lines = []
    lines.append("# OmniGuard-RAG — Path B: Embedding-Space Comparison Report")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
                 f"Compares TF-IDF (sparse, the representation Path A's results are built on) "
                 f"against LSA/TruncatedSVD (dense, {results_by_space['lsa']['dim']} components, fit on the "
                 f"same TF-IDF matrix). Mean ± 95% CI (Student's-t) across "
                 f"{len(seeds)} seed(s): {seeds}, {n_queries} queries/seed, docs_per_topic={docs_per_topic}.")
    lines.append("")
    lines.append("## 1. Does the existing calibration hold in LSA space? (measured, not assumed)")
    lines.append("")
    lines.append("Same clean-only top-5 retrieval measurement `risk_router.py`'s own comments cite "
                 "for `RISK_THRESHOLD`/`CONTENTION_THRESHOLD`, run against both spaces:")
    lines.append("")
    lines.append("| Embedding space | Clean cohesion floor (min) | Clean cohesion mean | "
                 "Current RISK_THRESHOLD | Margin | Clean contention (max) |")
    lines.append("|---|---|---|---|---|---|")
    for space in SPACES:
        cs = clean_stat_by_space[space]
        lines.append(f"| {space} | {cs['cohesion_min']:.3f} | {cs['cohesion_mean']:.3f} | "
                     f"{RISK_THRESHOLD} | {cs['cohesion_min'] - RISK_THRESHOLD:+.3f} | "
                     f"{cs['contention_max']:.3f} |")
    lines.append("")
    lines.append(f"Ring 1 (DRS) held-out false-positive rate: "
                 f"TF-IDF **{holdout_stat_by_space['tfidf']}%**, LSA **{holdout_stat_by_space['lsa']}%** "
                 f"(DRS recalibrates its own threshold from whichever reference embeddings it's given, "
                 f"so this number is a direct check on whether that self-recalibration worked in the new "
                 f"space, not something this script had to tune by hand).")
    lines.append("")
    lines.append("## 2. Main system comparison — TF-IDF vs. LSA, side by side")
    lines.append("")
    lines.append("| Defense Framework | Accuracy (TF-IDF) | Accuracy (LSA) | Overall ASR (TF-IDF) | "
                 "Overall ASR (LSA) | Stealth ASR (TF-IDF) | Stealth ASR (LSA) |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in MAIN_SYSTEMS:
        sa, sb = agg_main_by_space["tfidf"][name], agg_main_by_space["lsa"][name]
        lines.append(f"| {name} | {sa['accuracy']}% | {sb['accuracy']}% | {sa['overall_asr']}% | "
                     f"{sb['overall_asr']}% | {sa['stealth_asr']}% | {sb['stealth_asr']}% |")
    lines.append("")
    lines.append("## 3. Per-ring ablation ladder — TF-IDF vs. LSA, side by side")
    lines.append("")
    lines.append("| Ring configuration | Accuracy (TF-IDF) | Accuracy (LSA) | Overall ASR (TF-IDF) | "
                 "Overall ASR (LSA) |")
    lines.append("|---|---|---|---|---|")
    for name in ABLATION_VARIANTS:
        sa, sb = agg_abl_by_space["tfidf"][name], agg_abl_by_space["lsa"][name]
        lines.append(f"| {name} | {sa['accuracy']}% | {sb['accuracy']}% | {sa['overall_asr']}% | "
                     f"{sb['overall_asr']}% |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES)
    ap.add_argument("--docs-per-topic", type=int, default=30)
    ap.add_argument("--recalibrate-risk-threshold", action="store_true",
                     help="Override RISK_THRESHOLD with a value derived from this run's own measured "
                          "LSA cohesion floor, instead of the existing TF-IDF-measured constant.")
    args = ap.parse_args()

    if args.recalibrate_risk_threshold:
        import unified_rag_defense.risk_router as rr
        from unified_rag_defense.bench_common import build_world as _bw
        probe_world, _, _ = _bw(seed=args.seeds[0], docs_per_topic=args.docs_per_topic, embedding_space="lsa")
        probe_stats = measure_clean_statistics(probe_world, n_queries=args.n_queries, seed=args.seeds[0])
        new_threshold = round(probe_stats["cohesion_min"] - 0.01, 3)
        print(f"--recalibrate-risk-threshold: overriding RISK_THRESHOLD {rr.RISK_THRESHOLD} -> {new_threshold} "
              f"(measured LSA clean cohesion floor {probe_stats['cohesion_min']:.3f} minus a 0.01 margin)")
        rr.RISK_THRESHOLD = new_threshold
        import unified_rag_defense.ablations as abl
        abl.RISK_THRESHOLD = new_threshold

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    results_by_space, agg_main_by_space, agg_abl_by_space = {}, {}, {}
    holdout_stat_by_space, clean_stat_by_space = {}, {}

    for space in SPACES:
        print(f"Running embedding_space={space} ({len(args.seeds)} seed(s) x {args.n_queries} queries)...",
              flush=True)
        result = run_one_space(space, args.seeds, args.n_queries, args.docs_per_topic)
        world_probe, _, _ = build_world(seed=args.seeds[0], docs_per_topic=args.docs_per_topic,
                                         embedding_space=space)
        result["dim"] = world_probe.dim
        results_by_space[space] = result
        agg_main_by_space[space] = aggregate(result["main_raw"])
        agg_abl_by_space[space] = aggregate(result["ablation_raw"])
        holdout_stat_by_space[space] = str(summarize(result["holdout_fpr_per_seed"]))
        keys = result["clean_stats_per_seed"][0].keys()
        clean_stat_by_space[space] = {
            k: sum(d[k] for d in result["clean_stats_per_seed"]) / len(result["clean_stats_per_seed"])
            for k in keys
        }
        print(f"  done: dim={result['dim']}, holdout_fpr={holdout_stat_by_space[space]}%")

    print()
    print_side_by_side("MAIN SYSTEM COMPARISON -- TF-IDF vs. LSA", MAIN_SYSTEMS,
                        agg_main_by_space["tfidf"], agg_main_by_space["lsa"], "tfidf", "lsa")
    print()
    print_side_by_side("PER-RING ABLATION LADDER -- TF-IDF vs. LSA", ABLATION_VARIANTS,
                        agg_abl_by_space["tfidf"], agg_abl_by_space["lsa"], "tfidf", "lsa")
    print()
    print("Clean-only top-5 retrieval statistics (the measurement RISK_THRESHOLD/CONTENTION_THRESHOLD "
          "are justified against):")
    for space in SPACES:
        cs = clean_stat_by_space[space]
        print(f"  {space:>6}: cohesion floor={cs['cohesion_min']:.3f} (current RISK_THRESHOLD={RISK_THRESHOLD}, "
              f"margin={cs['cohesion_min']-RISK_THRESHOLD:+.3f})  contention max={cs['contention_max']:.3f} "
              f"(current CONTENTION_THRESHOLD={CONTENTION_THRESHOLD})")

    raw_out = {
        "seeds": args.seeds, "n_queries": args.n_queries, "docs_per_topic": args.docs_per_topic,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "recalibrated_risk_threshold": args.recalibrate_risk_threshold,
        "by_space": results_by_space,
    }
    (results_dir / "path_b_raw_results.json").write_text(json.dumps(raw_out, indent=2), encoding="utf-8")
    write_markdown_report(results_dir / "path_b_report.md", args.seeds, args.n_queries, args.docs_per_topic,
                          results_by_space, agg_main_by_space, agg_abl_by_space,
                          holdout_stat_by_space, clean_stat_by_space)
    print(f"\nWrote {results_dir / 'path_b_raw_results.json'}")
    print(f"Wrote {results_dir / 'path_b_report.md'}")


if __name__ == "__main__":
    main()
