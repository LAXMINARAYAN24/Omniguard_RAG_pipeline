"""
omniguard_production.claims — Proposition Extraction, NLI Contradiction Matrix & Multi-Signal Risk Routing.
"""
from .claim_extractor import AtomicClaim, ClaimExtractor
from .nli_verifier import NLIVerifier
from .risk_scorer import RoutingAction, CalibratedRiskRouter

__all__ = [
    "AtomicClaim",
    "ClaimExtractor",
    "NLIVerifier",
    "RoutingAction",
    "CalibratedRiskRouter",
]
