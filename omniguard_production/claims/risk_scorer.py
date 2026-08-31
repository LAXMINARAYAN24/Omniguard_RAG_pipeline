"""
risk_scorer.py — Continuous Multi-Signal Calibrated Risk Router (Ring 2).

Replaces brittle single-pair max() contradiction with weighted contradiction density
across relevant, independent evidence pairs and multi-signal calibrated risk attribution.
"""
from __future__ import annotations
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from .claim_extractor import AtomicClaim, ClaimExtractor
from .nli_verifier import NLIVerifier
from ..trust.provenance import ProductionChunk


class RoutingAction(str, Enum):
    SAFE_PASS = "SAFE_PASS"
    TARGETED_CONSENSUS = "TARGETED_CONSENSUS"
    QUARANTINE_BLOCK = "QUARANTINE_BLOCK"


class CalibratedRiskRouter:
    """
    Combines spectral SVD shifts, weighted NLI contradiction density,
    query security signals, and provenance trust scores into a calibrated continuous risk score.
    """

    def __init__(self,
                 w_drs: float = 0.35,
                 w_nli: float = 0.35,
                 w_query: float = 0.15,
                 w_prov: float = 0.15,
                 consensus_threshold: float = 0.30,
                 quarantine_threshold: float = 0.75,
                 top_k_weight: float = 0.60):
        self.w_drs = w_drs
        self.w_nli = w_nli
        self.w_query = w_query
        self.w_prov = w_prov
        self.consensus_threshold = consensus_threshold
        self.quarantine_threshold = quarantine_threshold
        self.top_k_weight = top_k_weight
        self.extractor = ClaimExtractor()
        self.nli = NLIVerifier()

    def _compute_claim_relevance_weight(self, claim_a: AtomicClaim, claim_b: AtomicClaim) -> float:
        """
        Computes relevance/subject overlap weight between two claims.
        Only claims discussing related topics/entities should contribute significantly to contradiction density.
        """
        if claim_a.subject and claim_b.subject:
            # Check subject token overlap
            s_a = set(re.findall(r"\w+", claim_a.subject.lower()))
            s_b = set(re.findall(r"\w+", claim_b.subject.lower()))
            if s_a & s_b:
                return 1.0

        # General token overlap
        t_a = set(re.findall(r"\w+", claim_a.text.lower()))
        t_b = set(re.findall(r"\w+", claim_b.text.lower()))
        overlap = t_a & t_b
        if not overlap:
            return 0.05

        return min(1.0, len(overlap) / max(1, min(len(t_a), len(t_b))))

    def compute_weighted_contradiction_density(self,
                                               claims: List[AtomicClaim],
                                               contradiction_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates:
        1. top_k_mean: mean of the top-3 most severe pairwise contradictions.
        2. weighted_density: relevance-weighted contradiction density across all cross-chunk pairs.
        3. effective_nli_intensity: calibrated combination of peak contradiction and contradiction density.
        """
        n = len(claims)
        if n < 2 or contradiction_matrix.size == 0:
            return 0.0, 0.0, 0.0

        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_scores = []
        weighted_sum = 0.0
        total_weight = 0.0

        for i, j in zip(upper_tri_indices[0], upper_tri_indices[1]):
            # Skip claims from the same source chunk
            if claims[i].source_chunk_id and claims[i].source_chunk_id == claims[j].source_chunk_id:
                continue

            c_score = float(contradiction_matrix[i, j])
            rel_weight = self._compute_claim_relevance_weight(claims[i], claims[j])

            pairwise_scores.append(c_score)
            weighted_sum += c_score * rel_weight
            total_weight += rel_weight

        if not pairwise_scores:
            return 0.0, 0.0, 0.0

        # Top-k mean contradiction (k=min(3, len(pairwise_scores)))
        sorted_scores = sorted(pairwise_scores, reverse=True)
        top_k = sorted_scores[:min(3, len(sorted_scores))]
        top_k_mean = float(np.mean(top_k)) if top_k else 0.0

        # Weighted density
        density = (weighted_sum / total_weight) if total_weight > 0.0 else 0.0

        # Effective intensity: convex combination of peak severity and distributed density
        effective_intensity = (self.top_k_weight * top_k_mean) + ((1.0 - self.top_k_weight) * density)
        effective_intensity = float(min(1.0, max(0.0, effective_intensity)))

        return round(top_k_mean, 4), round(density, 4), round(effective_intensity, 4)

    def evaluate_retrieval_set(self,
                               query_text: str,
                               chunks: List[ProductionChunk],
                               drs_shift_score: float = 0.0,
                               query_security_flags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculates multi-signal risk score across retrieved candidate chunks
        using weighted contradiction density and observable risk attribution.
        """
        if not chunks:
            return {
                "composite_risk_score": 0.0,
                "routing_action": RoutingAction.SAFE_PASS,
                "drs_score": drs_shift_score,
                "nli_contradiction_intensity": 0.0,
                "nli_top_k_contradiction": 0.0,
                "nli_contradiction_density": 0.0,
                "query_risk": 0.0,
                "provenance_risk": 0.0,
                "claims": [],
                "contradiction_matrix": [],
                "claim_contradiction_matrix": [],
                "risk_attribution": {},
                "routing_reasons": ["No candidate chunks retrieved"]
            }

        # 1. Extract atomic claims from all retrieved chunks
        all_claims: List[AtomicClaim] = []
        for c in chunks:
            all_claims.extend(self.extractor.extract_from_chunk(c))

        # 2. Compute NLI Full Relation Matrices (Entailment, Contradiction, Neutral)
        ent_matrix, c_matrix, _ = self.nli.compute_full_relation_matrices(all_claims)
        top_k_contra, contra_density, effective_nli_intensity = self.compute_weighted_contradiction_density(
            all_claims, c_matrix
        )

        # Compute Chunk-Level Contradiction Matrix from claim mapping
        n_chunks = len(chunks)
        chunk_contra_matrix = np.zeros((n_chunks, n_chunks), dtype=np.float64)

        if c_matrix.size > 0 and ent_matrix.size > 0:
            for i in range(n_chunks):
                c_id_i = chunks[i].chunk_id
                claim_indices_i = [idx for idx, cl in enumerate(all_claims) if cl.source_chunk_id == c_id_i]
                for j in range(i + 1, n_chunks):
                    c_id_j = chunks[j].chunk_id
                    claim_indices_j = [idx for idx, cl in enumerate(all_claims) if cl.source_chunk_id == c_id_j]

                    if not claim_indices_i or not claim_indices_j:
                        continue

                    max_c = 0.0
                    max_e = 0.0
                    for ci in claim_indices_i:
                        for cj in claim_indices_j:
                            c_score = float(c_matrix[ci, cj])
                            e_score = float(ent_matrix[ci, cj])
                            if c_score > max_c:
                                max_c = c_score
                            if e_score > max_e:
                                max_e = e_score

                    # Entailment discounting: if propositions strongly entail each other, contradiction is nullified
                    net_contra = max_c * (1.0 - max_e)
                    if max_e >= 0.70:
                        net_contra = 0.0

                    chunk_contra_matrix[i, j] = round(net_contra, 4)
                    chunk_contra_matrix[j, i] = round(net_contra, 4)

        # 3. Query Security Risk
        q_flags = query_security_flags or []
        query_risk = min(1.0, 0.40 * len(q_flags))

        # 4. Provenance & Trust Risk
        prov_penalties = []
        for c in chunks:
            penalty = 0.0
            if c.security_flags:
                penalty += 0.30 * len(c.security_flags)
            if c.trust_score < 0.8:
                penalty += (1.0 - c.trust_score)
            prov_penalties.append(penalty)

        avg_prov_risk = min(1.0, float(np.mean(prov_penalties))) if prov_penalties else 0.0

        # Compute Continuous Composite Risk
        composite_risk = (
            self.w_drs * min(1.0, max(0.0, drs_shift_score)) +
            self.w_nli * effective_nli_intensity +
            self.w_query * query_risk +
            self.w_prov * avg_prov_risk
        )
        composite_risk = min(1.0, max(0.0, composite_risk))

        # Compile Routing Reasons
        reasons = []
        if drs_shift_score > 0.4:
            reasons.append(f"High spectral DRS shift detected ({drs_shift_score:.2f})")
        if effective_nli_intensity > 0.4:
            reasons.append(f"Significant NLI claim contradiction intensity ({effective_nli_intensity:.2f})")
        if query_risk > 0.3:
            reasons.append(f"Adversarial query injection indicators present ({len(q_flags)} flags)")
        if avg_prov_risk > 0.3:
            reasons.append(f"Retrieved chunks exhibit degraded provenance trust ({avg_prov_risk:.2f})")
        if not reasons:
            reasons.append("Retrieval set exhibits high logical coherence and clean provenance")

        # Determine Action
        if composite_risk >= self.quarantine_threshold:
            action = RoutingAction.QUARANTINE_BLOCK
        elif composite_risk >= self.consensus_threshold or effective_nli_intensity >= 0.35:
            action = RoutingAction.TARGETED_CONSENSUS
        else:
            action = RoutingAction.SAFE_PASS

        return {
            "composite_risk_score": round(composite_risk, 4),
            "routing_action": action,
            "drs_score": round(drs_shift_score, 4),
            "nli_contradiction_intensity": effective_nli_intensity,
            "nli_top_k_contradiction": top_k_contra,
            "nli_contradiction_density": contra_density,
            "query_risk": round(query_risk, 4),
            "provenance_risk": round(avg_prov_risk, 4),
            "claim_count": len(all_claims),
            "claims": [c.text for c in all_claims],
            "contradiction_matrix": chunk_contra_matrix.tolist(),
            "claim_contradiction_matrix": c_matrix.tolist(),
            "risk_attribution": {
                "drs_weighted": round(self.w_drs * min(1.0, max(0.0, drs_shift_score)), 4),
                "nli_weighted": round(self.w_nli * effective_nli_intensity, 4),
                "query_weighted": round(self.w_query * query_risk, 4),
                "prov_weighted": round(self.w_prov * avg_prov_risk, 4),
            },
            "routing_reasons": reasons
        }
