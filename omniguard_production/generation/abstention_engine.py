"""
abstention_engine.py — 5-State Calibrated Abstention & Refusal Framework.

Manages graceful, calibrated abstention to eliminate hallucinated answers under uncertainty.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from ..trust.provenance import ProductionChunk
from ..consensus.lgo_analyzer import ConsensusStatus, GWCCDecision
from .citation_tracker import CitationAuditReport


class GenerationState(str, Enum):
    ANSWER = "ANSWER"
    PARTIAL_ANSWER = "PARTIAL_ANSWER"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    SECURITY_BLOCK = "SECURITY_BLOCK"


@dataclass
class AbstentionDecision:
    state: GenerationState
    final_output: str
    confidence: float
    reason: str
    can_proceed_to_generate: bool
    fallback_message: Optional[str] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)


class CalibratedAbstentionEngine:
    """Evaluates multi-ring telemetry to determine generation state and abstention actions."""

    def __init__(self,
                 min_retrieval_confidence: float = 0.40,
                 min_grounding_ratio: float = 0.50):
        self.min_retrieval_confidence = min_retrieval_confidence
        self.min_grounding_ratio = min_grounding_ratio

    def evaluate_pre_generation(self,
                                is_query_blocked: bool,
                                consensus_decision: Optional[GWCCDecision],
                                retrieved_chunks: List[ProductionChunk]) -> AbstentionDecision:
        """Determines if the system must abstain before calling the LLM."""
        if is_query_blocked:
            return AbstentionDecision(
                state=GenerationState.SECURITY_BLOCK,
                final_output="Request blocked by query security gateway due to adversarial prompt injection risks.",
                confidence=0.0,
                reason="Query flagged by security screener.",
                can_proceed_to_generate=False
            )

        if not retrieved_chunks:
            return AbstentionDecision(
                state=GenerationState.INSUFFICIENT_EVIDENCE,
                final_output="Information not available in provided sources.",
                confidence=0.0,
                reason="No relevant evidence chunks retrieved.",
                can_proceed_to_generate=False
            )

        if consensus_decision is not None:
            if consensus_decision.status == ConsensusStatus.CONFLICTING_POOLS:
                return AbstentionDecision(
                    state=GenerationState.CONFLICTING_EVIDENCE,
                    final_output=(
                        "Unable to provide a definitive answer: verified sources contain conflicting assertions. "
                        f"{consensus_decision.explanation}"
                    ),
                    confidence=consensus_decision.confidence_score,
                    reason="Contradictory evidence pools detected without consensus.",
                    can_proceed_to_generate=False
                )
            elif consensus_decision.status == ConsensusStatus.INSUFFICIENT_EVIDENCE:
                return AbstentionDecision(
                    state=GenerationState.INSUFFICIENT_EVIDENCE,
                    final_output="Information not available in provided sources.",
                    confidence=0.0,
                    reason="Evidence insufficient for consensus verification.",
                    can_proceed_to_generate=False
                )

        return AbstentionDecision(
            state=GenerationState.ANSWER,
            final_output="",
            confidence=consensus_decision.confidence_score if consensus_decision else 0.85,
            reason="Evidence validated; proceeding with grounded generation.",
            can_proceed_to_generate=True
        )

    def evaluate_post_generation(self,
                                 generated_text: str,
                                 citation_audit: CitationAuditReport,
                                 allowed_chunks: List[ProductionChunk]) -> AbstentionDecision:
        """Validates generated text against citation and grounding constraints."""
        if citation_audit.invalid_citations > 0 and citation_audit.valid_citations == 0:
            return AbstentionDecision(
                state=GenerationState.INSUFFICIENT_EVIDENCE,
                final_output="Information not available in verified sources.",
                confidence=0.10,
                reason="Generated response contained only invalid/fabricated citations.",
                can_proceed_to_generate=False
            )

        if citation_audit.grounding_ratio < self.min_grounding_ratio:
            return AbstentionDecision(
                state=GenerationState.PARTIAL_ANSWER,
                final_output=generated_text,
                confidence=0.50,
                reason=f"Partial answer: grounding ratio ({citation_audit.grounding_ratio:.2f}) below threshold.",
                can_proceed_to_generate=True
            )

        return AbstentionDecision(
            state=GenerationState.ANSWER,
            final_output=generated_text,
            confidence=0.95,
            reason="Fully grounded and citation-verified response.",
            can_proceed_to_generate=True
        )
