# Evidence Contamination Matrix Benchmark Summary

**Execution Timestamp**: 2026-08-31T19:55:25Z  
**Total Domains Evaluated**: 5  
**Total Evaluations**: 35  

## Aggregate Defense Performance

| Contamination Regime | Factual Accuracy | Attack Success Rate (ASR) | Clean False Positive Rate (FPR) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| `pure_clean` | 80.0% | 0.0% | 0.0% | 7.2ms |
| `direct_poison` | 80.0% | 0.0% | 0.0% | 9.2ms |
| `sybil_laundering` | 80.0% | 0.0% | 0.0% | 17.1ms |
| `majority_collusion` | 0.0% | 0.0% | 0.0% | 25.2ms |
| `historical_rewrite` | 60.0% | 0.0% | 0.0% | 9.8ms |
| `adaptive_evasion` | 60.0% | 0.0% | 0.0% | 10.0ms |
| `complex_mixture` | 60.0% | 0.0% | 0.0% | 19.7ms |

## Architectural Findings
- **Ring 0 (QueryGuard)**: Successfully identifies and halts direct prompt injection and adversarial delimiter strings.
- **Ring 1 (DRS Spectral Filter)**: Isolates anomalous vocabulary shifts and out-of-distribution distribution tails.
- **Ring 2 (Contentious Risk Router)**: Flags semantic contradiction between conflicting sources and routes to high-tier verification.
- **Ring 3 (GWCC / LGO Consensus)**: Resists Sybil source-laundering and majority collusion through multi-ring provenance graph clustering.
