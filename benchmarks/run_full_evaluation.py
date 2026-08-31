"""
run_full_evaluation.py — Comprehensive Multi-Seed Statistical Evaluation Suite.

Runs BOTH the 6-system comparison and the 4-step per-ring ablation ladder,
across multiple independent seeds (each seed = an independently generated
corpus, query set, DRS calibration split, and set of attacks), and reports
mean +/- 95% confidence interval across seeds for every metric, plus real
wall-clock latency alongside call-counts.

Usage:
    python benchmarks/run_full_evaluation.py                       # 8 seeds x 200 queries
    python benchmarks/run_full_evaluation.py --quick                # 3 seeds x 60 queries, fast iteration
    python benchmarks/run_full_evaluation.py --seeds 7 11 23 42     # custom seed list
    python benchmarks/run_full_evaluation.py --n-queries 100
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unified_rag_defense.bench_common import build_world, run_system, measure_holdout_fpr
from unified_rag_defense import baselines
from unified_rag_defense.omniguard_pipeline import run_omniguard
from unified_rag_defense.ablations import VARIANTS as ABLATION_VARIANTS, dispatch_ablation
from unified_rag_defense.stats_utils import summarize, Stat

MAIN_SYSTEMS = [
    "Vanilla RAG", "DRS Only (2025)", "ShieldRAG Only (2026)",
    "RAGuard / ZKIP (2026)", "TriShield (2026)", "OmniGuard-RAG (Ours)",
]

DEFAULT_SEEDS = [7, 11, 23, 41, 59, 79, 97, 113]
QUICK_SEEDS = [7, 11, 23]

METRICS = ["accuracy", "overall_asr", "pidp_asr", "collusion_asr",
           "stealth_asr", "silent_asr", "avg_calls", "avg_latency_ms"]


def dispatch_main(name, query, pool, world, drs, centroid, trust_store, rng):
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


def tally_to_metrics(t) -> Dict[str, float]:
    return {
        "accuracy": t.accuracy, "overall_asr": t.overall_asr,
        "pidp_asr": t.regime_asr("pidp"), "collusion_asr": t.regime_asr("collusion"),
        "stealth_asr": t.regime_asr("collusion_stealth"), "silent_asr": t.regime_asr("silent"),
        "avg_calls": t.avg_calls, "avg_latency_ms": t.avg_latency_ms,
    }


def run_all_seeds(seeds: List[int], n_queries: int, docs_per_topic: int) -> dict:
    """Returns {"main": {name: {metric: [values, one per seed]}},
                "ablation": {name: {metric: [values]}},
                "holdout_fpr": [values, one per seed]}"""
    raw_main = {name: defaultdict(list) for name in MAIN_SYSTEMS}
    raw_ablation = {name: defaultdict(list) for name in ABLATION_VARIANTS}
    holdout_fprs = []

    for si, seed in enumerate(seeds):
        t0 = time.perf_counter()
        world, drs, centroid = build_world(seed=seed, docs_per_topic=docs_per_topic, drs_seed=seed)
        queries = world.sample_queries(n_queries)
        holdout_fprs.append(100.0 * measure_holdout_fpr(world, drs))

        for name in MAIN_SYSTEMS:
            t = run_system(name, world, queries, drs, centroid, dispatch_main, seed=seed)
            for k, v in tally_to_metrics(t).items():
                raw_main[name][k].append(v)

        for name in ABLATION_VARIANTS:
            t = run_system(name, world, queries, drs, centroid, dispatch_ablation, seed=seed)
            for k, v in tally_to_metrics(t).items():
                raw_ablation[name][k].append(v)

        print(f"  seed {seed} ({si + 1}/{len(seeds)}) done in {time.perf_counter() - t0:.1f}s", flush=True)

    return {"main": raw_main, "ablation": raw_ablation, "holdout_fpr": holdout_fprs}


def load_checkpoint(results_dir: Path, n_queries: int, docs_per_topic: int):
    p = results_dir / "path_a_raw_results.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("n_queries") != n_queries or data.get("docs_per_topic") != docs_per_topic:
        return None
    return data


def aggregate(raw: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Stat]]:
    return {name: {metric: summarize(values) for metric, values in per_metric.items()}
            for name, per_metric in raw.items()}


def _row(name: str, s: Dict[str, Stat]) -> str:
    return (f"{name:<24}| {str(s['accuracy']):>10} | {str(s['overall_asr']):>12} | "
            f"{str(s['pidp_asr']):>10} | {str(s['collusion_asr']):>14} | "
            f"{str(s['stealth_asr']):>12} | {str(s['silent_asr']):>10}")


def print_results_table(title: str, names: List[str], agg: Dict[str, Dict[str, Stat]]):
    header = (f"{'System / Ring configuration':<24}| {'Accuracy':>10} | {'Overall ASR':>12} | "
              f"{'PIDP ASR':>10} | {'Collusion ASR':>14} | {'Stealth ASR':>12} | {'Silent ASR':>10}")
    print("=" * len(header))
    print(title.center(len(header)))
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name in names:
        print(_row(name, agg[name]))
    print("=" * len(header))


def print_compute_table(agg_main: Dict[str, Dict[str, Stat]], agg_abl: Dict[str, Dict[str, Stat]]):
    header = f"{'System / Ring configuration':<24}| {'Avg Calls':>14} | {'Avg Latency (ms)':>18}"
    print("=" * len(header))
    print("COMPUTE COST -- CALL COUNT vs. REAL WALL-CLOCK LATENCY".center(len(header)))
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name in MAIN_SYSTEMS:
        s = agg_main[name]
        print(f"{name:<24}| {str(s['avg_calls']):>14} | {str(s['avg_latency_ms']):>18}")
    print("-" * len(header))
    for name in ABLATION_VARIANTS:
        s = agg_abl[name]
        print(f"{name:<24}| {str(s['avg_calls']):>14} | {str(s['avg_latency_ms']):>18}")
    print("=" * len(header))


def write_markdown_report(path: Path, seeds, n_queries, docs_per_topic, holdout_stat,
                           agg_main, agg_abl):
    lines = []
    lines.append("# OmniGuard-RAG — Path A Evaluation Report")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
                 f"All figures are mean ± 95% confidence interval (Student's-t) across "
                 f"{len(seeds)} independent seeds: {seeds}. Each seed is an independently "
                 f"regenerated corpus, query set, and DRS calibration split (not just a "
                 f"reshuffle of the same data) at {n_queries} queries/seed, "
                 f"docs_per_topic={docs_per_topic}.")
    lines.append("")
    lines.append(f"Ring 1 (DRS) held-out false-positive rate on fresh, non-malicious docs: "
                 f"**{holdout_stat}%** (n=5 fresh docs/topic/seed).")
    lines.append("")
    lines.append("## 1. Main system comparison")
    lines.append("")
    lines.append("| Defense Framework | Accuracy | Overall ASR | PIDP ASR | Collusion ASR | Stealth ASR | Silent ASR |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in MAIN_SYSTEMS:
        s = agg_main[name]
        lines.append(f"| {name} | {s['accuracy']}% | {s['overall_asr']}% | {s['pidp_asr']}% | "
                     f"{s['collusion_asr']}% | {s['stealth_asr']}% | {s['silent_asr']}% |")
    lines.append("")
    lines.append("## 2. Per-ring ablation ladder")
    lines.append("")
    lines.append("Same seeds, same queries, same attacks as Table 1 -- each row adds one ring "
                 "to the previous row's pipeline. See `unified_rag_defense/ablations.py` for "
                 "exactly what each step does and does not include.")
    lines.append("")
    lines.append("| Ring configuration | Accuracy | Overall ASR | PIDP ASR | Collusion ASR | Stealth ASR | Silent ASR |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ABLATION_VARIANTS:
        s = agg_abl[name]
        lines.append(f"| {name} | {s['accuracy']}% | {s['overall_asr']}% | {s['pidp_asr']}% | "
                     f"{s['collusion_asr']}% | {s['stealth_asr']}% | {s['silent_asr']}% |")
    lines.append("")
    lines.append("## 3. Compute cost: call count vs. wall-clock latency")
    lines.append("")
    lines.append("Wall-clock time measures this simulation's own Python/NumPy orchestration "
                 "cost around each system's own logic (there is no real LLM in this benchmark's "
                 "loop -- see `baselines.py`'s RAGuard/ZKIP docstring). It is NOT a production "
                 "LLM-latency estimate. It IS a genuine, measured comparison of how expensive "
                 "each defense's own processing is, independent of whatever LLM calls its "
                 "`avg_calls` number represents -- and the two rankings disagree (see below), "
                 "which avg_calls alone could not have shown.")
    lines.append("")
    lines.append("| System / Ring configuration | Avg Calls | Avg Latency (ms) |")
    lines.append("|---|---|---|")
    for name in MAIN_SYSTEMS:
        s = agg_main[name]
        lines.append(f"| {name} | {s['avg_calls']} | {s['avg_latency_ms']} |")
    for name in ABLATION_VARIANTS:
        s = agg_abl[name]
        lines.append(f"| {name} | {s['avg_calls']} | {s['avg_latency_ms']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                     help=f"seed list (default: {DEFAULT_SEEDS})")
    ap.add_argument("--quick", action="store_true",
                     help=f"fast iteration mode: seeds={QUICK_SEEDS}, n-queries=60")
    ap.add_argument("--n-queries", type=int, default=None, help="queries per seed (default: 200, or 60 with --quick)")
    ap.add_argument("--docs-per-topic", type=int, default=30)
    args = ap.parse_args()

    seeds_requested = args.seeds or (QUICK_SEEDS if args.quick else DEFAULT_SEEDS)
    n_queries = args.n_queries or (60 if args.quick else 200)
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    checkpoint = load_checkpoint(results_dir, n_queries, args.docs_per_topic)
    if checkpoint is not None:
        already_done = list(checkpoint["seeds"])
        new_seeds = [s for s in seeds_requested if s not in already_done]
        raw_main = {name: defaultdict(list, checkpoint["main_raw"].get(name, {})) for name in MAIN_SYSTEMS}
        raw_ablation = {name: defaultdict(list, checkpoint["ablation_raw"].get(name, {})) for name in ABLATION_VARIANTS}
        holdout_fprs = list(checkpoint["holdout_fpr_per_seed"])
        seeds = list(already_done)
        if already_done:
            print(f"Resuming: {len(already_done)} seed(s) already computed at n_queries={n_queries}, "
                  f"docs_per_topic={args.docs_per_topic}: {already_done}")
    else:
        new_seeds = seeds_requested
        raw_main = {name: defaultdict(list) for name in MAIN_SYSTEMS}
        raw_ablation = {name: defaultdict(list) for name in ABLATION_VARIANTS}
        holdout_fprs = []
        seeds = []

    if new_seeds:
        print(f"Running {len(new_seeds)} new seed(s) x {n_queries} queries x 7 regimes x "
              f"{len(MAIN_SYSTEMS)} systems + {len(ABLATION_VARIANTS)} ablation variants: {new_seeds}")
        t_start = time.perf_counter()
        for si, seed in enumerate(new_seeds):
            t0 = time.perf_counter()
            world, drs, centroid = build_world(seed=seed, docs_per_topic=args.docs_per_topic, drs_seed=seed)
            queries = world.sample_queries(n_queries)
            holdout_fprs.append(100.0 * measure_holdout_fpr(world, drs))
            for name in MAIN_SYSTEMS:
                t = run_system(name, world, queries, drs, centroid, dispatch_main, seed=seed)
                for k, v in tally_to_metrics(t).items():
                    raw_main[name][k].append(v)
            for name in ABLATION_VARIANTS:
                t = run_system(name, world, queries, drs, centroid, dispatch_ablation, seed=seed)
                for k, v in tally_to_metrics(t).items():
                    raw_ablation[name][k].append(v)
            seeds.append(seed)
            print(f"  seed {seed} ({si + 1}/{len(new_seeds)} new) done in {time.perf_counter() - t0:.1f}s", flush=True)

            raw_out = {
                "seeds": seeds, "n_queries": n_queries, "docs_per_topic": args.docs_per_topic,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "holdout_fpr_per_seed": holdout_fprs,
                "main_raw": {k: dict(v) for k, v in raw_main.items()},
                "ablation_raw": {k: dict(v) for k, v in raw_ablation.items()},
            }
            (results_dir / "path_a_raw_results.json").write_text(json.dumps(raw_out, indent=2), encoding="utf-8")
        print(f"New seeds done in {time.perf_counter() - t_start:.1f}s.\n")
    else:
        print("All requested seeds already computed -- nothing new to run.\n")

    agg_main = aggregate(raw_main)
    agg_abl = aggregate(raw_ablation)
    holdout_stat = summarize(holdout_fprs)

    print(f"Ring 1 (DRS) held-out false-positive rate on fresh, non-malicious docs: {holdout_stat}%")
    print(f"(docs_per_topic={args.docs_per_topic}, n_queries={n_queries}/seed, seeds={seeds})")
    print()
    print_results_table("MAIN SYSTEM COMPARISON -- MEAN +/- 95% CI ACROSS SEEDS", MAIN_SYSTEMS, agg_main)
    print()
    print_results_table("PER-RING ABLATION LADDER -- MEAN +/- 95% CI ACROSS SEEDS", ABLATION_VARIANTS, agg_abl)
    print()
    print_compute_table(agg_main, agg_abl)

    write_markdown_report(results_dir / "path_a_report.md", seeds, n_queries, args.docs_per_topic,
                           holdout_stat, agg_main, agg_abl)
    print(f"\nWrote {results_dir / 'path_a_raw_results.json'}")
    print(f"Wrote {results_dir / 'path_a_report.md'}")


if __name__ == "__main__":
    main()
