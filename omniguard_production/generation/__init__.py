"""
omniguard_production.generation — Source-Anchored Prompting, Inline Citation Auditing, CoV & Calibrated Abstention.
"""
from .prompt_assembler import PromptAssembler, STRICT_GROUNDED_SYSTEM_PROMPT
from .citation_tracker import CitationTracker, CitationAuditReport
from .cov_engine import ChainOfVerificationEngine, CoVResult, CoVVerificationCheck
from .abstention_engine import GenerationState, AbstentionDecision, CalibratedAbstentionEngine

__all__ = [
    "PromptAssembler",
    "STRICT_GROUNDED_SYSTEM_PROMPT",
    "CitationTracker",
    "CitationAuditReport",
    "ChainOfVerificationEngine",
    "CoVResult",
    "CoVVerificationCheck",
    "GenerationState",
    "AbstentionDecision",
    "CalibratedAbstentionEngine",
]
