"""
omniguard_production — Enterprise Adaptive Evidence-Aware Security Control Plane for Production RAG (v2).

Features:
- Dual-Mode Architecture (Preserves deterministic research benchmarks alongside enterprise async control plane)
- Parser Sandbox & Pre-Ingestion Injection Screener
- Dense Neural + Okapi BM25 Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
- Neural Cross-Attention Reranking with Dynamic Adaptive-k
- Multi-Signal Calibrated Risk Routing (Ring 0, 1, 2)
- Graph-Theoretic Consensus & Leave-Group-Out Collusion Isolation (Ring 3: GWCC v2)
- Source-Anchored Prompting, Inline Citation Auditing & Chain-of-Verification (CoV)
- 5-State Calibrated Abstention Framework
- Multi-Tenant Persistent Trust & Reputation Ledger
- OpenTelemetry Distributed Tracing & Operational Metrics
"""

from .pipeline import OmniGuardProductionPipeline, PipelineExecutionResult
from .trust.provenance import DocumentState, DocumentMetadata, ProductionChunk, ProductionDocument
from .trust.trust_store import PersistentTrustStore, TrustEvent, TrustEventType
from .gateway.parser_sandbox import ParserSandbox
from .gateway.injection_screener import InjectionScreener
from .gateway.query_gateway import QuerySecurityGateway
from .retrieval.dense_retriever import DenseRetriever
from .retrieval.bm25_retriever import BM25Retriever
from .retrieval.hybrid_fusion import HybridFusion
from .retrieval.cross_reranker import CrossEncoderReranker
from .claims.claim_extractor import ClaimExtractor
from .claims.nli_verifier import NLIVerifier
from .claims.risk_scorer import CalibratedRiskRouter, RoutingAction
from .consensus.evidence_graph import EvidenceGraph
from .consensus.lgo_analyzer import LGOConsensusAnalyzer, ConsensusStatus, GWCCDecision
from .generation.prompt_assembler import PromptAssembler
from .generation.citation_tracker import CitationTracker, CitationAuditReport
from .generation.cov_engine import ChainOfVerificationEngine, CoVResult
from .generation.abstention_engine import CalibratedAbstentionEngine, GenerationState, AbstentionDecision
from .observability.tracer import PipelineTracer, TelemetrySpan
from .observability.metrics import ProductionMetricsCollector

__version__ = "2.0.0"

__all__ = [
    "OmniGuardProductionPipeline",
    "PipelineExecutionResult",
    "DocumentState",
    "DocumentMetadata",
    "ProductionChunk",
    "ProductionDocument",
    "PersistentTrustStore",
    "TrustEvent",
    "TrustEventType",
    "ParserSandbox",
    "InjectionScreener",
    "QuerySecurityGateway",
    "DenseRetriever",
    "BM25Retriever",
    "HybridFusion",
    "CrossEncoderReranker",
    "ClaimExtractor",
    "NLIVerifier",
    "CalibratedRiskRouter",
    "RoutingAction",
    "EvidenceGraph",
    "LGOConsensusAnalyzer",
    "ConsensusStatus",
    "GWCCDecision",
    "PromptAssembler",
    "CitationTracker",
    "CitationAuditReport",
    "ChainOfVerificationEngine",
    "CoVResult",
    "CalibratedAbstentionEngine",
    "GenerationState",
    "AbstentionDecision",
    "PipelineTracer",
    "TelemetrySpan",
    "ProductionMetricsCollector"
]
