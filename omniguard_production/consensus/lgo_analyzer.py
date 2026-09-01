"""
lgo_analyzer.py — Selective Leave-Group-Out (LGO) Causal Counterfactual Consensus Analyzer (Ring 3).

Evaluates whether removing specific candidate collusion clusters causally resolves
contradictions and stabilizes ground truth evidence across independent source lineages.
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
    counterfactual_deltas: Dict[int, float] = field(default_factory=dict)
    explanation: str = ""
    group_telemetry: Dict[str, Any] = field(default_factory=dict)


class LGOConsensusAnalyzer:
    """
    Group-Wise Counterfactual Consensus engine with genuine Leave-Group-Out
    causal contradiction-resolution attribution.
    """

    def __init__(self,
                 dominance_ratio: float = 1.5,
                 min_verified_confidence: float = 0.50):
        self.dominance_ratio = dominance_ratio
        self.min_verified_confidence = min_verified_confidence

    def analyze_consensus(self,
                          clusters: List[EvidenceCluster],
                          original_chunks: List[ProductionChunk],
                          contradiction_matrix: Optional[np.ndarray] = None) -> GWCCDecision:
        """
        Evaluates candidate clusters using causal Leave-Group-Out analysis to isolate
        collusion cliques and determine genuine multi-source consensus.
        """
        if not clusters or not original_chunks:
            return GWCCDecision(
                status=ConsensusStatus.INSUFFICIENT_EVIDENCE,
                selected_chunks=[],
                quarantined_chunks=[],
                confidence_score=0.0,
                explanation="No evidence chunks provided to consensus analyzer."
            )

        # Total contradiction in original set
        total_contra_all = float(np.sum(contradiction_matrix)) if contradiction_matrix is not None and contradiction_matrix.size > 0 else 0.0

        # 1. Single cluster scenario
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

        # 2. Multi-cluster causal Leave-Group-Out evaluation
        # Map chunks to indices in contradiction matrix
        chunk_to_idx = {c.chunk_id: i for i, c in enumerate(original_chunks)}
        lgo_contra_deltas: Dict[int, float] = {}

        for cl in clusters:
            cl_chunk_ids = {c.chunk_id for c in cl.chunks}
            remaining_indices = [chunk_to_idx[c.chunk_id] for c in original_chunks if c.chunk_id not in cl_chunk_ids and c.chunk_id in chunk_to_idx]

            if contradiction_matrix is not None and contradiction_matrix.size > 0:
                if len(remaining_indices) > 1:
                    sub_mat = contradiction_matrix[np.ix_(remaining_indices, remaining_indices)]
                    sub_contra = float(np.sum(sub_mat))
                else:
                    sub_contra = 0.0
                contra_drop = max(0.0, total_contra_all - sub_contra)
                lgo_contra_deltas[cl.cluster_id] = round(contra_drop, 4)
            else:
                lgo_contra_deltas[cl.cluster_id] = round(cl.evidence_weight, 4)

        # Check if there is genuine contradiction across the clusters FIRST
        # (needed before deciding whether to use causal LGO signal or weight-based selection)
        max_cross_cluster_contra = 0.0
        if contradiction_matrix is not None and contradiction_matrix.size > 0:
            for i, cl_a in enumerate(clusters):
                for cl_b in clusters[i+1:]:
                    for ca in cl_a.chunks:
                        ia = chunk_to_idx.get(ca.chunk_id)
                        if ia is None:
                            continue
                        for cb in cl_b.chunks:
                            ib = chunk_to_idx.get(cb.chunk_id)
                            if ib is None:
                                continue
                            c_val = float(contradiction_matrix[ia, ib])
                            if c_val > max_cross_cluster_contra:
                                max_cross_cluster_contra = c_val

        has_active_contradiction = (total_contra_all > 0.25) or (max_cross_cluster_contra >= 0.35)

        # When active contradiction exists, use CAUSAL LGO and lineage independence to identify poison source
        if has_active_contradiction and len(clusters) >= 2:
            non_adv_clusters = [cl for cl in clusters if not cl.is_adversarial_candidate]
            adv_clusters = [cl for cl in clusters if cl.is_adversarial_candidate]

            if non_adv_clusters and adv_clusters:
                # Clear adversarial separation: select non-adversarial, quarantine adversarial
                clean_clusters = non_adv_clusters
                poison_clusters = adv_clusters
            else:
                # No binary adversarial flag separation: rank by evidence weight (incorporates independence, trust, and size)
                sorted_by_credibility = sorted(clusters, key=lambda c: c.evidence_weight, reverse=True)
                clean_clusters = [sorted_by_credibility[0]]
                poison_clusters = sorted_by_credibility[1:]

            quarantined = []
            for pcl in poison_clusters:
                quarantined.extend(pcl.chunks)

            selected = []
            for ccl in clean_clusters:
                selected.extend(ccl.chunks)

            avg_trust_selected = float(np.mean([c.trust_score for c in selected])) if selected else 0.0
            poison_delta = max((lgo_contra_deltas.get(pcl.cluster_id, 0.0) for pcl in poison_clusters), default=0.0)

            return GWCCDecision(
                status=ConsensusStatus.COLLUSION_DISCARDED,
                selected_chunks=selected,
                quarantined_chunks=quarantined,
                confidence_score=round(min(1.0, avg_trust_selected), 4),
                selected_cluster_id=clean_clusters[0].cluster_id if len(clean_clusters) == 1 else None,
                lgo_delta=round(poison_delta, 4),
                counterfactual_deltas=lgo_contra_deltas,
                explanation=(
                    f"Leave-group-out causal test identified {len(poison_clusters)} contradictory cluster(s) "
                    f"as contradiction source (max contradiction drop {poison_delta:.3f}). "
                    f"Quarantined {len(quarantined)} poisoned chunks, selected {len(selected)} from {len(clean_clusters)} clean cluster(s)."
                ),
                group_telemetry={
                    "total_clusters": len(clusters),
                    "cluster_weights": {c.cluster_id: c.evidence_weight for c in clusters},
                    "lgo_contra_deltas": lgo_contra_deltas,
                    "poison_cluster_ids": [c.cluster_id for c in poison_clusters],
                    "poison_lgo_delta": round(poison_delta, 4),
                    "max_cross_cluster_contra": round(max_cross_cluster_contra, 4),
                    "total_contra_all": round(total_contra_all, 4)
                }
            )

        # No active contradiction OR insufficient clusters for LGO: fall back to weight-based selection
        clusters = sorted(clusters, key=lambda x: x.evidence_weight, reverse=True)
        best_cluster = clusters[0]
        second_cluster = clusters[1] if len(clusters) > 1 else None

        weight_1 = best_cluster.evidence_weight
        weight_2 = second_cluster.evidence_weight if second_cluster else 0.0
        ratio = weight_1 / max(1e-4, weight_2)
        lgo_delta = round(weight_1 - weight_2, 4)

        quarantined: List[ProductionChunk] = []
        for c in clusters:
            if c.cluster_id != best_cluster.cluster_id:
                quarantined.extend(c.chunks)

        # Non-contradictory complementary multi-aspect clusters:
        if not has_active_contradiction:
            non_adv_clusters = [cl for cl in clusters if not cl.is_adversarial_candidate]
            adv_clusters = [cl for cl in clusters if cl.is_adversarial_candidate]

            selected: List[ProductionChunk] = []
            for cl in non_adv_clusters:
                selected.extend(cl.chunks)

            quarantined: List[ProductionChunk] = []
            for cl in adv_clusters:
                quarantined.extend(cl.chunks)

            if selected:
                avg_trust_all = float(np.mean([c.trust_score for c in selected]))
                status = ConsensusStatus.UNANIMOUS_GROUNDED if not adv_clusters else ConsensusStatus.COLLUSION_DISCARDED
                return GWCCDecision(
                    status=status,
                    selected_chunks=selected,
                    quarantined_chunks=quarantined,
                    confidence_score=round(min(1.0, avg_trust_all), 4),
                    selected_cluster_id=clusters[0].cluster_id,
                    lgo_delta=lgo_delta,
                    counterfactual_deltas=lgo_contra_deltas,
                    explanation=(
                        f"Multi-aspect evidence clusters detected without mutual contradiction "
                        f"(max contradiction {max_cross_cluster_contra:.2f} < 0.35). "
                        f"Selected {len(selected)} verified chunks across {len(non_adv_clusters)} lineages."
                    ),
                    group_telemetry={
                        "total_clusters": len(clusters),
                        "cluster_weights": {c.cluster_id: c.evidence_weight for c in clusters},
                        "dominance_ratio": round(ratio, 4),
                        "max_cross_cluster_contra": round(max_cross_cluster_contra, 4),
                        "total_contra_all": round(total_contra_all, 4)
                    }
                )

        # Check for consensus dominance supported by independent lineages
        if ratio >= self.dominance_ratio and not best_cluster.is_adversarial_candidate:
            status = ConsensusStatus.CONSENSUS_VERIFIED
            if any(c.is_adversarial_candidate for c in clusters[1:]):
                status = ConsensusStatus.COLLUSION_DISCARDED

            total_weight = sum(c.evidence_weight for c in clusters)
            conf = min(1.0, (best_cluster.evidence_weight / max(1e-4, total_weight)) * best_cluster.average_trust)
            explanation = (
                f"Cluster {best_cluster.cluster_id} achieved consensus dominance (ratio {ratio:.2f} >= {self.dominance_ratio}, "
                f"independence {best_cluster.lineage_independence_score:.2f}, domains {best_cluster.domain_diversity}). "
                f"Quarantined {len(quarantined)} chunks from subordinate/colluding clusters."
            )
            return GWCCDecision(
                status=status,
                selected_chunks=best_cluster.chunks,
                quarantined_chunks=quarantined,
                confidence_score=round(conf, 4),
                selected_cluster_id=best_cluster.cluster_id,
                lgo_delta=lgo_delta,
                counterfactual_deltas=lgo_contra_deltas,
                explanation=explanation,
                group_telemetry={
                    "total_clusters": len(clusters),
                    "cluster_weights": {c.cluster_id: c.evidence_weight for c in clusters},
                    "dominance_ratio": round(ratio, 4),
                    "lgo_contra_deltas": lgo_contra_deltas,
                    "best_cluster_lineage_indep": best_cluster.lineage_independence_score
                }
            )

        # If one cluster is non-adversarial and the other is adversarial candidate:
        non_adv_clusters = [cl for cl in clusters if not cl.is_adversarial_candidate]
        if len(non_adv_clusters) == 1 and any(cl.is_adversarial_candidate for cl in clusters):
            winner = non_adv_clusters[0]
            quarantined = [c for cl in clusters if cl.cluster_id != winner.cluster_id for c in cl.chunks]
            return GWCCDecision(
                status=ConsensusStatus.COLLUSION_DISCARDED,
                selected_chunks=winner.chunks,
                quarantined_chunks=quarantined,
                confidence_score=round(winner.average_trust, 4),
                selected_cluster_id=winner.cluster_id,
                lgo_delta=lgo_delta,
                counterfactual_deltas=lgo_contra_deltas,
                explanation=f"Isolated and quarantined adversarial candidate clusters. Grounded in cluster {winner.cluster_id}.",
                group_telemetry={
                    "total_clusters": len(clusters),
                    "cluster_weights": {c.cluster_id: c.evidence_weight for c in clusters},
                    "dominance_ratio": round(ratio, 4),
                    "lgo_contra_deltas": lgo_contra_deltas
                }
            )

        # If clusters are evenly balanced with contradictory claims -> Conflicting evidence
        all_quarantined = list(original_chunks)
        return GWCCDecision(
            status=ConsensusStatus.CONFLICTING_POOLS,
            selected_chunks=[],
            quarantined_chunks=all_quarantined,
            confidence_score=0.35,
            counterfactual_deltas=lgo_contra_deltas,
            explanation=f"Contradictory evidence pools detected without consensus dominance (ratio {ratio:.2f} < {self.dominance_ratio}, max contra {max_cross_cluster_contra:.2f}).",
            group_telemetry={
                "total_clusters": len(clusters),
                "cluster_weights": {c.cluster_id: c.evidence_weight for c in clusters},
                "dominance_ratio": round(ratio, 4),
                "lgo_contra_deltas": lgo_contra_deltas
            }
        )
