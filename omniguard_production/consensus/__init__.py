"""
omniguard_production.consensus — Graph-Theoretic Evidence Network & Group-Wise Counterfactual Consensus (GWCC v2).
"""
from .evidence_graph import EvidenceCluster, EvidenceGraph
from .lgo_analyzer import ConsensusStatus, GWCCDecision, LGOConsensusAnalyzer

__all__ = [
    "EvidenceCluster",
    "EvidenceGraph",
    "ConsensusStatus",
    "GWCCDecision",
    "LGOConsensusAnalyzer",
]
