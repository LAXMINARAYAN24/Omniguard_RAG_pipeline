# OmniGuard-RAG — Manual: Running the Complete Project on Your Desktop

*Comprehensive guide for executing the OmniGuard-RAG 4-ring defense pipeline, benchmarks, and interactive web dashboard.*

---

## 0. Prerequisites & Requirements

- **Python 3.9 or newer** (tested and confirmed working on Python 3.12).
- **Core dependencies**: `numpy`, `scikit-learn`, `scipy`.
- **Optional local LLM integration**: Ollama (`http://localhost:11434`), LM Studio (`http://localhost:1234/v1`), or the built-in offline neural synthesizer (zero external dependencies).

**Check your Python version:**
```bash
python --version
```

---

## 1. Directory Structure Overview

The repository is cleanly structured into modular components:

```
Practical_Training/
├── README.md                      # Project documentation & quickstart
├── run_dashboard.py               # Interactive Web Dashboard entry point
├── unified_rag_defense/           # Core 4-ring defense architecture & baselines
│   ├── query_guard.py             # Ring 0: Query-Path Repetition Ratio & Suffix Guard
│   ├── drs_filter.py              # Ring 1: Spectral SVD Directional Relative Shifts
│   ├── risk_router.py             # Ring 2: Semantic Cohesion & Contention Risk Router
│   ├── gwcc_consensus.py          # Ring 3: Group-Wise Counterfactual Consensus (LOO + LGO)
│   ├── omniguard_pipeline.py      # End-to-End 4-Ring Pipeline & Dynamic Trust Store
│   ├── baselines.py               # Literature Baselines (Vanilla, DRS, ShieldRAG, RAGuard, TriShield)
│   └── attack_simulator.py        # 6 Poison Attack Generators (Standard, PIDP, Collusion, Stealth, Silent)
├── dashboard/                     # Web Dashboard backend & Single Page App frontend
│   ├── dashboard_server.py        # Multi-threaded HTTP server with security controls
│   ├── llm_client.py              # Local LLM connector (Ollama, LM Studio, Built-in)
│   ├── rag_defense_engine.py      # Live telemetry & attack sandbox engine
│   └── static/                    # Frontend UI (index.html, styles.css, app.js)
├── benchmarks/                    # Evaluation & diagnostic benchmarking suite
│   ├── run_benchmark.py           # Unified CLI dispatcher for benchmarks & evaluation
│   ├── run_omniguard_benchmark.py # 6-system x 7-regime standard benchmark
│   ├── run_full_evaluation.py     # Multi-seed statistically rigorous evaluation suite
│   ├── run_embedding_comparison.py# TF-IDF vs SentenceTransformer embedding comparison
│   ├── run_gwcc_diagnostic.py     # Ring 3 LOO vs LGO diagnostic test suite
│   └── automate_verification.py   # Automated full-pipeline verification runner
├── docs/                          # Project documentation, research papers & reports
│   ├── RUN_MANUAL.md              # Detailed execution & troubleshooting guide
│   ├── COMPREHENSIVE_BENCHMARK_REPORT.md # Empirical findings & architecture specifications
│   ├── OmniGuard_RAG_Report.md    # Complete academic report
│   ├── PROJECT_COMPLETION_SUMMARY.md # Milestone summary
│   ├── walkthrough.md             # Guided step-by-step walkthrough
│   └── literature/                # 6 base research papers (summaries & PDFs)
├── results/                       # Multi-seed raw benchmark JSONs & markdown reports
└── tests/                         # Security audit & dashboard integration test suites
```

---

## 2. Virtual Environment & Dependencies

Set up an isolated virtual environment:

```bash
# Windows PowerShell:
python -m venv venv
venv\Scripts\Activate.ps1

# macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

Install core dependencies:
```bash
pip install numpy scikit-learn scipy
```

---

## 3. Interactive Web Dashboard

Launch the web studio to interactively test retrieval, attack injections, and defense telemetry:

```bash
python run_dashboard.py
```
This automatically starts the server at `http://127.0.0.1:8000` and opens your browser.

**Features:**
- **Streaming Simulation**: Interactive prompt submission with real-time defense stage visualization.
- **7 Poisoning Regimes**: Clean, Keyword Stuffing, PIDP Injection, Collusion ($k=2,3$), Stealth CATG, and Silent Subspace.
- **Deep Telemetry Drawer**: Live inspection of Ring 0 repetition ratio, Ring 1 DRS eigenvalues, Ring 2 contention scores, and Ring 3 counterfactual consensus.
- **Multi-System Comparison Matrix**: Side-by-side comparison of Vanilla RAG vs DRS vs ShieldRAG vs RAGuard vs TriShield vs OmniGuard-RAG.
- **Local LLM Integration**: Connect to Ollama, LM Studio, or use the built-in offline engine.

---

## 4. Running Benchmarks via Unified CLI

Use `benchmarks/run_benchmark.py` to run any benchmark suite:

### 4a. Main 6-System Standard Benchmark (~35s)
```bash
python benchmarks/run_benchmark.py --suite omniguard
```
Evaluates all 6 systems against 7 attack regimes under fixed seed.

### 4b. GWCC Ring 3 Diagnostic (~5s)
```bash
python benchmarks/run_benchmark.py --suite diagnostic
```
Sweeps collusion sizes ($k=3, 5, 8, 12$) to verify Leave-Group-Out (LGO) consensus against malicious cliques.

### 4c. Full Multi-Seed Evaluation (8 seeds × 200 queries)
```bash
# Quick validation (3 seeds, 60 queries/seed, ~45s):
python benchmarks/run_benchmark.py --suite full --quick

# Full publication-grade evaluation (8 seeds, 200 queries/seed, ~7 min):
python benchmarks/run_benchmark.py --suite full
```
Computes mean $\pm$ 95% confidence intervals via Student's-t distribution and saves `results/path_a_report.md` and `results/path_a_raw_results.json`.

### 4d. Automated Full-Pipeline Verification
```bash
python benchmarks/run_benchmark.py --suite verify
```
Runs test suites, benchmarks, and generates timestamped audit reports in `results/`.

---

## 5. Security & Test Suites

Run integration and security audit test suites:

```bash
# Dashboard API integration tests:
python -m unittest tests/test_dashboard_api.py

# Security audit suite (SSRF, Rate Limiting, Security Headers, CORS, Payload Bounds):
python -m unittest tests/test_security_audit.py
```
