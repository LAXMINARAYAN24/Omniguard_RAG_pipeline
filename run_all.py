"""
run_all.py — Unified Master Automation Runner for OmniGuard-RAG

Executes the entire verification suite sequentially:
1. Security and Unit Tests (Tests Suite)
2. Track B: Real-Inference 4-vs-1 Majority Collusion Experiment
3. Track B: Real-Inference End-to-End Multi-Domain Production Evaluation
4. Track A: Standard 6-System Benchmark
5. Diagnostic: Ring 3 GWCC Consensus Divergence Diagnostic
6. Diagnostic: Sparse TF-IDF vs. Dense Embedding Space Comparison

Usage:
    python run_all.py
"""

import sys
import os
import time
import unittest
from pathlib import Path

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f"  🚀  {title.upper()}")
    print("=" * 80 + "\n")


def run_unit_tests():
    print_banner("1. Running Unit & Integration Test Suites")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("[!] Unit test suite encountered failures.")
        return False
    print("\n[✓] All unit & integration tests passed successfully.\n")
    return True


def run_track_b_majority_collusion():
    print_banner("2. Track B: 4-vs-1 Majority Collusion Experiment")
    from evaluation.real_inference.run_majority_collusion_experiment import run_majority_collusion_evaluation
    results = run_majority_collusion_evaluation(backend="auto")
    return bool(results)


def run_track_b_production_eval():
    print_banner("3. Track B: Real-Inference Multi-Domain Production Evaluation")
    from evaluation.real_inference.run_production_eval import run_track_b_evaluation
    results = run_track_b_evaluation(backend="auto")
    return bool(results)


def run_track_a_benchmark():
    print_banner("4. Track A: Standard 6-System Controlled Benchmark")
    from benchmarks.run_omniguard_benchmark import run_omniguard_benchmark
    results = run_omniguard_benchmark(seed=42, n_queries=30)
    return bool(results)


def run_gwcc_diagnostic():
    print_banner("5. Diagnostic: Ring 3 GWCC Consensus Divergence")
    from benchmarks.run_gwcc_diagnostic import run_gwcc_diagnostic
    res = run_gwcc_diagnostic()
    return res.get("passed", True) if isinstance(res, dict) else True


def run_embedding_comparison():
    print_banner("6. Diagnostic: Sparse TF-IDF vs. Dense Embedding Space")
    from benchmarks.run_embedding_comparison import run_embedding_comparison
    res = run_embedding_comparison()
    return bool(res)


def main():
    start_time = time.time()
    print("*" * 80)
    print("  OMNIGUARD-RAG MASTER VERIFICATION & EVALUATION RUNNER")
    print("  Version: Version_ONE Production & Research Dual-Track")
    print("*" * 80)

    stages = [
        ("Unit & Integration Tests", run_unit_tests),
        ("Track B: Majority Collusion Experiment", run_track_b_majority_collusion),
        ("Track B: Production Evaluation", run_track_b_production_eval),
        ("Track A: Standard 6-System Benchmark", run_track_a_benchmark),
        ("Ring 3 GWCC Diagnostic", run_gwcc_diagnostic),
        ("Embedding Space Comparison", run_embedding_comparison)
    ]

    summary = []
    for stage_name, stage_fn in stages:
        t0 = time.time()
        try:
            success = stage_fn()
            duration = time.time() - t0
            summary.append((stage_name, "PASSED" if success else "FAILED", f"{duration:.2f}s"))
        except Exception as e:
            duration = time.time() - t0
            print(f"[!] Exception during {stage_name}: {e}")
            summary.append((stage_name, f"ERROR ({type(e).__name__})", f"{duration:.2f}s"))

    total_duration = time.time() - start_time
    print("\n" + "=" * 80)
    print("  MASTER AUTOMATION RUN SUMMARY")
    print("=" * 80)
    for name, status, dur in summary:
        print(f"  - {name:<45} [{status}] ({dur})")
    print("=" * 80)
    print(f"  Total Time Elapsed: {total_duration:.2f}s")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
