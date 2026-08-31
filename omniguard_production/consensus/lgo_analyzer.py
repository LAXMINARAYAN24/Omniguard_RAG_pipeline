"""
lgo_analyzer.py — Selective Leave-Group-Out (LGO) Counterfactual Consensus Analyzer (Ring 3).

Evaluates whether removing specific candidate collusion clusters resolves contradictions
and preserves stable ground truth evidence without combinatorial explosion.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
from .evidence_graph import EvidenceCluster, EvidenceGraph
from ..trust.provenance import ProductionChunk


class ConsensusStatus(str, Enum):
    UNANIMOUS_GROUNDED = "UNANIMOUS_GROUNDED"
    CONSENSUS_VERIFIED = "CONSENSUS_VERIFIED"
    COLLUSION_DISCARDED = "COLLUSION_DISCARDED"
    CONFLICTING_POOLS = "CONFLICTING_POOLS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class GWCCDecision:
    """Outcome of Group-Wise Counterfactual Consensus evaluation."""
    status: ConsensusStatus
    selected_chunks: List[ProductionChunk]
    quarantined_chunks: List[ProductionChunk]
    confidence_score: float
    selected_cluster_id: Optional[int] = None
    lgo_delta: float = 0.0
    explanation: str = ""
    group_telemetry: Dict[str, Any] = field(default_factory=dict)


class LGOConsensusAnalyzer:
    """Group-Wise Counterfactual Consensus engine with Leave-Group-Out isolation."""

    def __init__(self,
                 dominance_ratio: float = 1.6,
                 min_verified_confidence: float = 0.55):
        self.dominance_ratio = dominance_ratio
        self.min_verified_confidence = min_verified_confidence

    def analyze_consensus(self,
                          clusters: List[EvidenceCluster],
                          original_chunks: List[ProductionChunk]) -> GWCCDecision:
        """Evaluates clusters using selective Leave-Group-Out to isolate collusion and extract consensus."""
        if not clusters or not original_chunks:
            return GWCCDecision(
                status=ConsensusStatus.INSUFFICIENT_EVIDENCE,
                selected_chunks=[],
                quarantined_chunks=[],
                confidence_score=0.0,
                explanation="No evidence chunks provided to consensus analyzer."
            )

        # Single cluster scenario
        if len(clusters) == 1:
            c = clusters[0]
            if c.is_adversarial_candidate:
                return GWCCDecision(
                    status=ConsensusStatus.COLLUSION_DISCARDED,
                    selected_chunks=[],
                    quarantined_chunks=c.chunks,
                    confidence_score=0.20,
                    explanation="Single candidate cluster detected as adversarial/low-trust."
                )
            return GWCCDecision(
                status=ConsensusStatus.UNANIMOUS_GROUNDED,
                selected_chunks=c.chunks,
                quarantined_chunks=[],
                confidence_score=min(1.0, c.average_trust),
                selected_cluster_id=c.cluster_id,
                explanation="All retrieved chunks form a single unanimous, non-contradictory support cluster."
            )

        # Multi-cluster scenario: inspect relative evidence weights and apply Leave-Group-Out
        total_weight = sum(c.evidence_weight for c in clusters)
        best_cluster = clusters[0]
        second_cluster = clusters[1] if len(clusters) > 1 else None

        # Counterfactual: Leave-Group-Out for best_cluster vs second_cluster
        weight_1 = best_cluster.evidence_weight
        weight_2 = second_cluster.evidence_weight if second_cluster else 0.0

        ratio = weight_1 / max(1e-4, weight_2)
        lgo_delta = round(weight_1 - weight_2, 4)

        quarantined: List[ProductionChunk] = []
        for c in clusters:
            if c.cluster_id != best_cluster.cluster_id:
                quarantined.extend(c.chunks)

        # Check for clear consensus dominance
        if ratio >= self.dominance_ratio and not best_cluster.is_adversarial_candidate:
            status = ConsensusStatus.CONSENSUS_VERIFIED
            if any(c.is_adversarial_candidate for c in clusters[1:]):
                status = ConsensusStatus.COLLUSION_DISCARDED

            conf = min(1.0, (best_cluster.evidence_weight / max(1e-4, total_weight)) * best_cluster.average_trust)
            explanation = (
                f"Cluster {best_cluster.cluster_id} achieved consensus dominance (ratio {ratio:.2f} >= {self.dominance_ratio}). "
                f"Quarantined {len(quarantined)} chunks from subordinate/conflicting clusters."
            )
            return GWCCDecision(
                status=status,
                selected_chunks=best_cluster.chunks,
                quarantined_chunks=quarantined,
                confidence_score=round(conf, 4),
                selected_cluster_id=best_cluster.cluster_id,
                lgo_delta=lgo_delta,
                explanation=explanation,
                group_telemetry={
                    "total_clusters": len(clusters),
                    "cluster_weights": {c.cluster_id: c.evidence_weight for c in clusters},
                    "dominance_ratio": round(ratio, 4)
                }
            )

        # If clusters are evenly balanced with contradictory claims -> Conflicting evidence
        all_quarantined = list(original_chunks)
        return GWCCDecision(
            status=ConsensusStatus.CONFLICTING_POOLS,
            selected_chunks=[],
            quarantined_chunks=all_quarantined,
            confidence_score=0.35,
            explanation=f"Contradictory evidence pools detected without consensus dominance (ratio {ratio:.2f} < {self.dominance_ratio}).",
            group_telemetry={
                "total_clusters": len(clusters),
                "cluster_weights": {c.cluster_id: c.evidence_weight for c in clusters},
                "dominance_ratio": round(ratio, 4)
            }
        )
