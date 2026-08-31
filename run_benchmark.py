"""
run_benchmark.py — Unified Benchmark & Evaluation CLI Dispatcher (Root Entry Point).
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.run_benchmark import main

if __name__ == "__main__":
    main()
