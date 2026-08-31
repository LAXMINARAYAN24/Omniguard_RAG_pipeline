"""
metrics.py — Real-time Evaluation & Operational Metrics Collector.

Calculates Attack Success Rate (ASR), Citation Precision, Latency Percentiles, and Quarantine Ratios.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict


class ProductionMetricsCollector:
    """Aggregates execution metrics across multiple queries and benchmark runs."""

    def __init__(self):
        self.latencies_ms: List[float] = []
        self.ring_quarantines: Dict[str, int] = defaultdict(int)
        self.total_queries: int = 0
        self.blocked_queries: int = 0
        self.citation_precisions: List[float] = []
        self.citation_recalls: List[float] = []
        self.grounding_ratios: List[float] = []

    def record_query(self,
                     total_latency_ms: float,
                     is_blocked: bool,
                     quarantined_ring: Optional[str] = None,
                     citation_precision: Optional[float] = None,
                     citation_recall: Optional[float] = None,
                     grounding_ratio: Optional[float] = None):
        """Records a single end-to-end query execution telemetry point."""
        self.total_queries += 1
        self.latencies_ms.append(total_latency_ms)

        if is_blocked:
            self.blocked_queries += 1

        if quarantined_ring:
            self.ring_quarantines[quarantined_ring] += 1

        if citation_precision is not None:
            self.citation_precisions.append(citation_precision)
        if citation_recall is not None:
            self.citation_recalls.append(citation_recall)
        if grounding_ratio is not None:
            self.grounding_ratios.append(grounding_ratio)

    def compute_summary(self) -> Dict[str, Any]:
        """Calculates percentiles and aggregate telemetry."""
        if not self.latencies_ms:
            return {"total_queries": 0}

        lats = np.array(self.latencies_ms)
        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))

        avg_prec = float(np.mean(self.citation_precisions)) if self.citation_precisions else 1.0
        avg_rec = float(np.mean(self.citation_recalls)) if self.citation_recalls else 1.0
        avg_ground = float(np.mean(self.grounding_ratios)) if self.grounding_ratios else 1.0

        return {
            "total_queries": self.total_queries,
            "blocked_queries": self.blocked_queries,
            "block_rate": round(self.blocked_queries / max(1, self.total_queries), 4),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_p99_ms": round(p99, 2),
            "avg_citation_precision": round(avg_prec, 4),
            "avg_citation_recall": round(avg_rec, 4),
            "avg_grounding_ratio": round(avg_ground, 4),
            "quarantines_by_ring": dict(self.ring_quarantines)
        }

    def get_summary(self) -> Dict[str, Any]:
        """Alias for compute_summary."""
        return self.compute_summary()
