"""
models.py — Universal Data Models, Types, and State Definitions for OmniGuard Production Control Plane.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from omniguard_production.trust.provenance import (
    DocumentState,
    DocumentMetadata,
    ProductionChunk,
    ProductionDocument
)
from omniguard_production.generation.abstention_engine import GenerationState

# Aliases for evaluation and benchmark compatibility
DefenseState = GenerationState


@dataclass
class QueryResult:
    """Convenience data container for query responses and benchmark evaluation."""
    query: str
    decision_state: DefenseState
    answer_text: str
    confidence: float
    citations: List[Dict[str, Any]] = field(default_factory=list)
    verified_chunks: List[Any] = field(default_factory=list)
    quarantined_chunks: List[Any] = field(default_factory=list)
    evidence_graph: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    route: str = "STANDARD_PASS"
    ring_telemetry: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "DocumentState",
    "DocumentMetadata",
    "ProductionChunk",
    "ProductionDocument",
    "GenerationState",
    "DefenseState",
    "QueryResult"
]
