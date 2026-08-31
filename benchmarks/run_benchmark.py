"""
benchmarks/run_benchmark.py — Unified Benchmark & Evaluation CLI for OmniGuard-RAG

Provides a single command-line interface to run any benchmark suite, diagnostic,
or verification pipeline in the repository.

Usage:
    python benchmarks/run_benchmark.py --suite omniguard        # Standard 6-system baseline benchmark
    python benchmarks/run_benchmark.py --suite full             # Multi-seed statistical evaluation (8 seeds x 200 queries)
    python benchmarks/run_benchmark.py --suite full --quick     # Fast multi-seed evaluation (3 seeds x 60 queries)
    python benchmarks/run_benchmark.py --suite diagnostic       # GWCC Ring 3 LOO vs. LGO diagnostic
    python benchmarks/run_benchmark.py --suite embedding        # TF-IDF vs. LSA embedding space comparison
    python benchmarks/run_benchmark.py --suite pipeline         # Complete automated verification pipeline
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Resolve repository root
BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUITE_MAP = {
    "omniguard": "benchmarks/run_omniguard_benchmark.py",
    "full": "benchmarks/run_full_evaluation.py",
    "diagnostic": "benchmarks/run_gwcc_diagnostic.py",
    "embedding": "benchmarks/run_embedding_comparison.py",
    "pipeline": "benchmarks/automate_verification.py",
    "verify": "benchmarks/automate_verification.py",
    "production": "evaluation/real_inference/run_production_eval.py",
    "real": "evaluation/real_inference/run_production_eval.py",
    "majority_collusion": "evaluation/real_inference/run_majority_collusion_experiment.py",
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified Benchmark & Evaluation CLI for OmniGuard-RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Suites:
  omniguard          Track A: 6-System empirical comparison on 7 attack regimes (controlled seed)
  full               Track A: Statistically rigorous multi-seed evaluation (mean ± 95% CI)
  production / real  Track B: Real-inference production evaluation (real corpora, real LLM, zero shortcuts)
  majority_collusion Track B: 4 Colluding shadow domains vs. 1 independent authority & clean same-domain FPR
  diagnostic         GWCC Ring 3 consensus divergence diagnostic
  embedding          Sparse TF-IDF vs. dense LSA embedding space comparison
  pipeline           Automated verification pipeline with git tracking & reports
"""
    )
    parser.add_argument(
        "--suite", "-s",
        choices=list(SUITE_MAP.keys()),
        default="omniguard",
        help="Benchmark suite to execute (default: omniguard)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run in quick iteration mode (for 'full' or 'pipeline' suites)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Custom seed list for multi-seed runs"
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        help="Number of queries per seed"
    )

    args, remaining_args = parser.parse_known_args()

    script_rel_path = SUITE_MAP[args.suite]
    script_path = REPO_ROOT / script_rel_path

    if not script_path.exists():
        print(f"[ERROR] Benchmark script not found at {script_path}")
        sys.exit(1)

    cmd = [sys.executable, str(script_path)]
    if args.quick:
        cmd.append("--quick")
    if args.seeds:
        cmd.append("--seeds")
        cmd.extend(str(s) for s in args.seeds)
    if args.n_queries:
        cmd.extend(["--n-queries", str(args.n_queries)])
    cmd.extend(remaining_args)

    print("=" * 72)
    print(f" 🚀 Executing Suite: '{args.suite}' ({script_rel_path})")
    print(f" 🔧 Command: {' '.join(cmd)}")
    print("=" * 72 + "\n")

    try:
        res = subprocess.run(cmd, cwd=str(REPO_ROOT))
        sys.exit(res.returncode)
    except KeyboardInterrupt:
        print("\n[!] Benchmark interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
