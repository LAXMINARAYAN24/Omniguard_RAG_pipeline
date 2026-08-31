"""
risk_scorer.py — Continuous Multi-Signal Calibrated Risk Router (Ring 2).

Replaces binary heuristic routing with a multi-signal risk calibration model.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional
from enum import Enum
from .claim_extractor import AtomicClaim, ClaimExtractor
from .nli_verifier import NLIVerifier
from ..trust.provenance import ProductionChunk


class RoutingAction(str, Enum):
    SAFE_PASS = "SAFE_PASS"
    TARGETED_CONSENSUS = "TARGETED_CONSENSUS"
    QUARANTINE_BLOCK = "QUARANTINE_BLOCK"


class CalibratedRiskRouter:
    """Combines spectral SVD shifts, NLI contradiction intensity, and provenance trust signals."""

    def __init__(self,
                 w_drs: float = 0.35,
                 w_nli: float = 0.35,
                 w_query: float = 0.15,
                 w_prov: float = 0.15,
                 consensus_threshold: float = 0.30,
                 quarantine_threshold: float = 0.75):
        self.w_drs = w_drs
        self.w_nli = w_nli
        self.w_query = w_query
        self.w_prov = w_prov
        self.consensus_threshold = consensus_threshold
        self.quarantine_threshold = quarantine_threshold
        self.extractor = ClaimExtractor()
        self.nli = NLIVerifier()

    def evaluate_retrieval_set(self,
                               query_text: str,
                               chunks: List[ProductionChunk],
                               drs_shift_score: float = 0.0,
                               query_security_flags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculates multi-signal risk score across retrieved candidate chunks."""
        if not chunks:
            return {
                "composite_risk_score": 0.0,
                "routing_action": RoutingAction.SAFE_PASS,
                "drs_score": drs_shift_score,
                "nli_contradiction_intensity": 0.0,
                "query_risk": 0.0,
                "provenance_risk": 0.0,
                "claims": [],
                "contradiction_matrix": []
            }

        # 1. Extract atomic claims from all retrieved chunks
        all_claims: List[AtomicClaim] = []
        for c in chunks:
            all_claims.extend(self.extractor.extract_from_chunk(c))

        # 2. Compute NLI Contradiction Matrix
        c_matrix = self.nli.compute_contradiction_matrix(all_claims)
        nli_intensity = 0.0
        if c_matrix.size > 0 and len(all_claims) > 1:
            # Maximum off-diagonal contradiction or top-k mean contradiction
            upper_tri = c_matrix[np.triu_indices_from(c_matrix, k=1)]
            if len(upper_tri) > 0:
                nli_intensity = float(np.max(upper_tri))

        # Compute Chunk-Level Contradiction Matrix from claim mapping
        n_chunks = len(chunks)
        chunk_contra_matrix = np.zeros((n_chunks, n_chunks), dtype=np.float64)
        chunk_id_to_idx = {c.chunk_id: idx for idx, c in enumerate(chunks)}

        if c_matrix.size > 0:
            for i_cl, claim_i in enumerate(all_claims):
                c_idx_i = chunk_id_to_idx.get(claim_i.source_chunk_id)
                if c_idx_i is None:
                    continue
                for j_cl, claim_j in enumerate(all_claims):
                    c_idx_j = chunk_id_to_idx.get(claim_j.source_chunk_id)
                    if c_idx_j is None or c_idx_i == c_idx_j:
                        continue
                    score = float(c_matrix[i_cl, j_cl])
                    if score > chunk_contra_matrix[c_idx_i, c_idx_j]:
                        chunk_contra_matrix[c_idx_i, c_idx_j] = score

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
            self.w_nli * min(1.0, max(0.0, nli_intensity)) +
            self.w_query * query_risk +
            self.w_prov * avg_prov_risk
        )
        composite_risk = min(1.0, max(0.0, composite_risk))

        # Determine Action
        if composite_risk >= self.quarantine_threshold:
            action = RoutingAction.QUARANTINE_BLOCK
        elif composite_risk >= self.consensus_threshold:
            action = RoutingAction.TARGETED_CONSENSUS
        else:
            action = RoutingAction.SAFE_PASS

        return {
            "composite_risk_score": round(composite_risk, 4),
            "routing_action": action,
            "drs_score": round(drs_shift_score, 4),
            "nli_contradiction_intensity": round(nli_intensity, 4),
            "query_risk": round(query_risk, 4),
            "provenance_risk": round(avg_prov_risk, 4),
            "claim_count": len(all_claims),
            "claims": [c.text for c in all_claims],
            "contradiction_matrix": chunk_contra_matrix.tolist(),
            "claim_contradiction_matrix": c_matrix.tolist()
        }
