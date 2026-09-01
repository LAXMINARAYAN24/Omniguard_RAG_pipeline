"""
pipeline.py — End-to-End Enterprise RAG Control Plane Pipeline.

Orchestrates the complete 4-Ring defense lifecycle:
  Ingestion -> Query Gateway (Ring 0) -> Hybrid Retrieval (Dense+BM25) ->
  Spectral DRS (Ring 1) -> Claim NLI Risk Routing (Ring 2) ->
  GWCC v2 Graph Consensus (Ring 3) -> CoV & Verifiable Citations ->
  Calibrated Abstention -> Persistent Trust Ledger.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np

# Trust & Provenance
from .trust.provenance import (
    DocumentState,
    DocumentMetadata,
    ProductionChunk,
    ProductionDocument
)
from .trust.trust_store import (
    PersistentTrustStore,
    TrustEvent,
    TrustEventType
)

# Gateways
from .gateway.parser_sandbox import ParserSandbox
from .gateway.injection_screener import InjectionScreener
from .gateway.query_gateway import QuerySecurityGateway

# Embeddings & Retrieval
from .embeddings.base import EmbeddingProvider
from .embeddings.neural_provider import DenseNeuralEmbeddingProvider
from .embeddings.drs_engine import DRSEngine, DRSModel, DRSConfig, DRSScoreResult
from .embeddings.density_normalizer import DensityNormalizer
from .retrieval.dense_retriever import DenseRetriever
from .retrieval.bm25_retriever import BM25Retriever
from .retrieval.hybrid_fusion import HybridFusion
from .retrieval.cross_reranker import CrossEncoderReranker

# Claims & Consensus
from .claims.claim_extractor import ClaimExtractor
from .claims.nli_verifier import NLIVerifier
from .claims.risk_scorer import CalibratedRiskRouter, RoutingAction
from .consensus.evidence_graph import EvidenceGraph
from .consensus.lgo_analyzer import LGOConsensusAnalyzer, ConsensusStatus, GWCCDecision

# Generation & Verifiability
from .generation.prompt_assembler import PromptAssembler
from .generation.citation_tracker import CitationTracker, CitationAuditReport
from .generation.cov_engine import ChainOfVerificationEngine, CoVResult
from .generation.abstention_engine import (
    CalibratedAbstentionEngine,
    GenerationState,
    AbstentionDecision
)

# Observability
from .observability.tracer import PipelineTracer
from .observability.metrics import ProductionMetricsCollector


@dataclass
class PipelineExecutionResult:
    query: str
    generation_state: GenerationState
    answer_text: str
    confidence: float
    citations: CitationAuditReport
    verified_chunks: List[ProductionChunk]
    quarantined_chunks: List[ProductionChunk]
    cov_result: Optional[CoVResult] = None
    ring_telemetry: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)

    @property
    def decision_state(self) -> GenerationState:
        return self.generation_state

    @property
    def latency_ms(self) -> float:
        return float(self.trace.get("total_duration_ms", 0.0))

    @property
    def evidence_graph(self) -> Dict[str, Any]:
        return {
            "quarantined_chunks": self.quarantined_chunks,
            "verified_chunks": self.verified_chunks
        }

    @property
    def route(self) -> str:
        r2 = self.ring_telemetry.get("ring_2_risk")
        if isinstance(r2, dict):
            return str(r2.get("routing_action", "STANDARD_PASS"))
        return "STANDARD_PASS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "generation_state": self.generation_state.value,
            "answer_text": self.answer_text,
            "confidence": round(float(self.confidence), 4),
            "citations": {
                "total_citations": self.citations.total_citations,
                "valid_citations": self.citations.valid_citations,
                "invalid_citations": self.citations.invalid_citations,
                "citation_precision": round(float(self.citations.citation_precision), 4),
                "citation_recall": round(float(self.citations.citation_recall), 4),
                "grounding_ratio": round(float(self.citations.grounding_ratio), 4),
                "citation_entailment_precision": round(float(self.citations.citation_entailment_precision), 4),
                "is_fully_grounded": self.citations.is_fully_grounded
            },
            "verified_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "clean_text": c.clean_text,
                    "trust_score": round(float(c.trust_score), 4),
                    "publisher_domain": c.metadata.publisher_domain,
                    "title": c.metadata.title,
                    "source_id": c.metadata.source_id,
                    "tenant_id": c.metadata.tenant_id
                } for c in self.verified_chunks
            ],
            "quarantined_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "clean_text": c.clean_text,
                    "trust_score": round(float(c.trust_score), 4),
                    "publisher_domain": c.metadata.publisher_domain,
                    "title": c.metadata.title,
                    "security_flags": c.security_flags,
                    "tenant_id": c.metadata.tenant_id
                } for c in self.quarantined_chunks
            ],
            "cov_result": {
                "baseline_response": self.cov_result.baseline_response,
                "revised_response": self.cov_result.revised_response,
                "unsupported_claims_removed": self.cov_result.unsupported_claims_removed,
                "grounding_score": round(float(self.cov_result.grounding_score), 4),
                "corroboration_ratio": round(float(self.cov_result.corroboration_ratio), 4),
                "verification_checks": [
                    {
                        "question": check.question,
                        "target_claim": check.target_claim,
                        "verification_answer": check.verification_answer,
                        "is_supported": check.is_supported,
                        "supporting_chunk_id": check.supporting_chunk_id,
                        "supporting_domain": check.supporting_domain,
                        "entailment_score": check.entailment_score,
                        "contradiction_score": check.contradiction_score,
                        "corroborating_domains": check.corroborating_domains
                    }
                    for check in self.cov_result.verification_checks
                ],
                "telemetry": self.cov_result.telemetry
            } if self.cov_result else None,
            "ring_telemetry": self.ring_telemetry,
            "trace": self.trace
        }


class OmniGuardProductionPipeline:
    """The enterprise RAG security control plane."""

    def __init__(self,
                 embedding_provider: Optional[EmbeddingProvider] = None,
                 llm_generator_fn: Optional[Callable[[str, str], str]] = None,
                 tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self.embedding_provider = embedding_provider or DenseNeuralEmbeddingProvider()
        self.llm_generator_fn = llm_generator_fn

        # Gateways
        self.parser_sandbox = ParserSandbox()
        self.injection_screener = InjectionScreener()
        self.query_gateway = QuerySecurityGateway(self.injection_screener)

        # Ring 1: Spectral SVD DRS Engine
        self.drs_engine = DRSEngine()

        # Retrieval
        self.dense_retriever = DenseRetriever(self.embedding_provider)
        self.bm25_retriever = BM25Retriever()
        self.hybrid_fusion = HybridFusion(rrf_k=60, dense_weight=0.6, sparse_weight=0.4)
        self.cross_reranker = CrossEncoderReranker(embedding_provider=self.embedding_provider)

        # Claims & Consensus
        self.claim_extractor = ClaimExtractor()
        self.nli_verifier = NLIVerifier()
        self.risk_router = CalibratedRiskRouter()
        self.evidence_graph = EvidenceGraph(embedding_provider=self.embedding_provider)
        self.lgo_analyzer = LGOConsensusAnalyzer()

        # Generation & Abstention
        self.prompt_assembler = PromptAssembler()
        self.citation_tracker = CitationTracker(nli_verifier=self.nli_verifier)
        self.cov_engine = ChainOfVerificationEngine(
            llm_generate_fn=self.llm_generator_fn,
            claim_extractor=self.claim_extractor,
            nli_verifier=self.nli_verifier
        )
        self.abstention_engine = CalibratedAbstentionEngine()

        # Trust Ledger & Metrics
        self.trust_store = PersistentTrustStore()
        self.metrics_collector = ProductionMetricsCollector()

    def calibrate_drs(self, clean_chunks: List[ProductionChunk]):
        """Fits SVD low-variance directions and calibrates threshold on clean reference embeddings."""
        clean_embs = []
        for c in clean_chunks:
            if c.embedding is not None:
                clean_embs.append(c.embedding)
            else:
                emb = self.embedding_provider.embed_text(c.clean_text)
                c.embedding = emb
                clean_embs.append(emb)

        if clean_embs:
            arr = np.array(clean_embs, dtype=np.float64)
            self.drs_engine.calibrate_from_embeddings(arr)

    def ingest_document(self,
                        raw_text: str,
                        metadata: Optional[DocumentMetadata] = None,
                        doc_id: Optional[str] = None) -> ProductionDocument:
        """Ingests, sanitizes, security-screens, chunks, embeds, and indexes a document."""
        meta = metadata or DocumentMetadata(tenant_id=self.tenant_id)
        doc = self.parser_sandbox.process_document(raw_text, doc_id=doc_id, metadata=meta)

        # Pre-index security screen on all chunks
        for chunk in doc.chunks:
            # Embed chunk text
            chunk.embedding = self.embedding_provider.embed_text(chunk.clean_text)

            screen_report = self.injection_screener.screen_text(chunk.clean_text)
            if screen_report["is_suspicious"]:
                chunk.security_flags.extend(screen_report["matched_flags"])
                chunk.state = DocumentState.SUSPICIOUS
                chunk.trust_score = max(0.1, chunk.trust_score - 0.40)
                # Record bounded hierarchical penalty in trust store
                self.trust_store.record_hierarchical_penalty(
                    tenant_id=meta.tenant_id,
                    publisher_domain=meta.publisher_domain,
                    source_id=meta.source_id,
                    document_id=doc.doc_id,
                    content_hash=chunk.content_hash,
                    reason=f"Malicious pattern in chunk {chunk.chunk_id}: {screen_report['matched_flags']}",
                    base_penalty=-0.40
                )
            else:
                chunk.state = DocumentState.SCANNED
                # Update trust with effective reputation from ledger
                rep = self.trust_store.get_effective_trust(
                    tenant_id=meta.tenant_id,
                    publisher_domain=meta.publisher_domain,
                    source_id=meta.source_id,
                    content_hash=chunk.content_hash
                )
                chunk.trust_score = rep["composite_trust"]

        # Index valid chunks into dense and BM25 retrievers
        valid_chunks = [c for c in doc.chunks if c.state != DocumentState.SUSPICIOUS]
        if valid_chunks:
            self.dense_retriever.index_chunks(valid_chunks)
            self.bm25_retriever.index_chunks(self.dense_retriever.chunks)
            for c in valid_chunks:
                c.state = DocumentState.INDEXED

            # Auto-calibrate DRS on initial clean corpus if uncalibrated
            if not self.drs_engine.is_calibrated() and len(self.dense_retriever.chunks) >= 6:
                self.calibrate_drs(self.dense_retriever.chunks)

        return doc

    def query(self,
              query_text: str,
              top_k: int = 10,
              tenant_id: Optional[str] = None,
              enable_cov: bool = True) -> PipelineExecutionResult:
        """Executes the full end-to-end grounded query pipeline."""
        tracer = PipelineTracer()
        active_tenant = tenant_id or self.tenant_id
        quarantined_chunks: List[ProductionChunk] = []

        # =========================================================================
        # Span 1: Ring 0 — Query Security Gateway
        # =========================================================================
        span_q = tracer.start_span("ring_0_query_gateway")
        q_report = self.query_gateway.inspect_query(query_text)
        is_query_blocked = q_report["is_injection_blocked"]
        cleaned_query = q_report["cleaned_query"]
        span_q.finish(
            status="FLAGGED" if q_report["security_flags"] else "OK",
            injection_risk=q_report["injection_risk"],
            is_suffix_detected=q_report["is_suffix_detected"],
            flags=q_report["security_flags"]
        )

        if is_query_blocked:
            decision = self.abstention_engine.evaluate_pre_generation(
                is_query_blocked=True,
                consensus_decision=None,
                retrieved_chunks=[]
            )
            trace_data = tracer.finish_trace()
            self.metrics_collector.record_query(
                total_latency_ms=trace_data["total_duration_ms"],
                is_blocked=True,
                quarantined_ring="Ring_0_Query"
            )
            return PipelineExecutionResult(
                query=query_text,
                generation_state=decision.state,
                answer_text=decision.final_output,
                confidence=0.0,
                citations=CitationAuditReport(0, 0, 0, 0.0, 0.0, 0.0),
                verified_chunks=[],
                quarantined_chunks=[],
                ring_telemetry={"ring_0": q_report},
                trace=trace_data
            )

        # =========================================================================
        # Span 2: Hybrid Retrieval & Neural Reranking
        # =========================================================================
        span_ret = tracer.start_span("hybrid_retrieval")
        dense_hits = self.dense_retriever.search(cleaned_query, top_k=top_k * 2, tenant_id=active_tenant)
        sparse_hits = self.bm25_retriever.search(cleaned_query, top_k=top_k * 2, tenant_id=active_tenant)

        fused_hits = self.hybrid_fusion.fuse_rrf(dense_hits, sparse_hits, top_k=top_k * 2)
        candidates = [c for c, _ in fused_hits]

        reranked = self.cross_reranker.rerank(cleaned_query, candidates)
        adaptive_hits = self.cross_reranker.select_adaptive_k(reranked)
        selected_candidates = [c for c, _, _ in adaptive_hits]
        span_ret.finish(candidates_count=len(selected_candidates))

        # =========================================================================
        # Span 3: Ring 1 & Ring 2 — Calibrated Multi-Signal Risk Routing & DRS
        # =========================================================================
        span_risk = tracer.start_span("ring_1_2_risk_routing")

        # Evaluate spectral SVD Directional Relative Shifts (Ring 1)
        drs_report: DRSScoreResult = self.drs_engine.evaluate_retrieval_set(selected_candidates)
        drs_shift_score = drs_report.max_drs_score / max(1.0, drs_report.threshold) if drs_report.is_calibrated else 0.10
        drs_shift_score = float(min(1.0, max(0.0, drs_shift_score)))

        risk_report = self.risk_router.evaluate_retrieval_set(
            query_text=cleaned_query,
            chunks=selected_candidates,
            drs_shift_score=drs_shift_score,
            query_security_flags=q_report["security_flags"]
        )
        routing_action = risk_report["routing_action"]
        span_risk.finish(
            routing_action=routing_action,
            composite_risk=risk_report["composite_risk_score"],
            nli_intensity=risk_report["nli_contradiction_intensity"],
            drs_spectral_anomaly=drs_report.is_spectral_anomaly_detected
        )

        # =========================================================================
        # Span 4: Ring 3 — Graph-Theoretic GWCC v2 Consensus with Lineage Gating
        # =========================================================================
        span_gwcc = tracer.start_span("ring_3_gwcc_consensus")
        consensus_decision: Optional[GWCCDecision] = None
        verified_chunks: List[ProductionChunk] = selected_candidates

        if routing_action in {RoutingAction.TARGETED_CONSENSUS, RoutingAction.QUARANTINE_BLOCK}:
            contra_mat = np.array(risk_report["contradiction_matrix"]) if risk_report["contradiction_matrix"] else None
            G = self.evidence_graph.build_graph(selected_candidates, contradiction_matrix=contra_mat)
            clusters = self.evidence_graph.detect_communities(G)
            consensus_decision = self.lgo_analyzer.analyze_consensus(clusters, selected_candidates, contradiction_matrix=contra_mat)

            verified_chunks = consensus_decision.selected_chunks
            quarantined_chunks.extend(consensus_decision.quarantined_chunks)

            # Record hierarchical trust ledger updates for quarantined chunks
            for q_chunk in consensus_decision.quarantined_chunks:
                self.trust_store.record_hierarchical_penalty(
                    tenant_id=active_tenant,
                    publisher_domain=q_chunk.metadata.publisher_domain,
                    source_id=q_chunk.metadata.source_id,
                    document_id=q_chunk.doc_id,
                    content_hash=q_chunk.content_hash,
                    reason=f"Isolated during GWCC Ring 3 consensus evaluation: {consensus_decision.explanation}",
                    base_penalty=-0.35
                )

            span_gwcc.finish(
                status=consensus_decision.status,
                selected_count=len(verified_chunks),
                quarantined_count=len(quarantined_chunks),
                lgo_delta=consensus_decision.lgo_delta
            )
        else:
            span_gwcc.finish(status="BYPASS_SAFE_PASS", selected_count=len(verified_chunks))

        # Check pre-generation abstention conditions
        pre_decision = self.abstention_engine.evaluate_pre_generation(
            is_query_blocked=False,
            consensus_decision=consensus_decision,
            retrieved_chunks=verified_chunks
        )

        if not pre_decision.can_proceed_to_generate:
            trace_data = tracer.finish_trace()
            self.metrics_collector.record_query(
                total_latency_ms=trace_data["total_duration_ms"],
                is_blocked=True,
                quarantined_ring="Ring_3_Consensus" if consensus_decision else "Retrieval"
            )
            return PipelineExecutionResult(
                query=query_text,
                generation_state=pre_decision.state,
                answer_text=pre_decision.final_output,
                confidence=pre_decision.confidence,
                citations=CitationAuditReport(0, 0, 0, 0.0, 0.0, 0.0),
                verified_chunks=verified_chunks,
                quarantined_chunks=quarantined_chunks,
                ring_telemetry={
                    "ring_0": q_report,
                    "ring_1_drs": drs_report.__dict__,
                    "ring_2_risk": risk_report,
                    "ring_3_gwcc": consensus_decision.__dict__ if consensus_decision else None
                },
                trace=trace_data
            )

        # =========================================================================
        # Span 5: Source-Anchored Prompting & Generation
        # =========================================================================
        span_gen = tracer.start_span("generation_and_cov")
        prompt_payload = self.prompt_assembler.assemble_prompt(cleaned_query, verified_chunks)

        if self.llm_generator_fn is not None:
            raw_response = self.llm_generator_fn(prompt_payload["system_prompt"], prompt_payload["user_prompt"])
        else:
            # High-fidelity built-in grounded synthesizer
            top_c = verified_chunks[0]
            title = top_c.metadata.title or top_c.doc_id
            h_short = top_c.content_hash[:8]
            raw_response = (
                f"{top_c.clean_text} "
                f"[Doc: {title} | Chunk: {top_c.chunk_index} | Hash: {h_short}]"
            )

        # Execute Chain-of-Verification (CoV) with independent corroboration pool
        cov_res = None
        final_response = raw_response
        if enable_cov and verified_chunks:
            # The corroboration pool includes verified chunks + other indexed chunks for cross-lineage checking
            corroboration_pool = list(self.dense_retriever.chunks) if self.dense_retriever.chunks else verified_chunks
            cov_res = self.cov_engine.run_verification(
                query_text=cleaned_query,
                baseline_response=raw_response,
                verified_chunks=verified_chunks,
                independent_corroboration_pool=corroboration_pool
            )
            final_response = cov_res.revised_response

        # Citation Tracking & Grounding Audit with NLI Entailment
        citation_audit = self.citation_tracker.audit_response(
            generated_text=final_response,
            allowed_chunks=verified_chunks,
            verify_semantic_entailment=True
        )
        post_decision = self.abstention_engine.evaluate_post_generation(
            generated_text=final_response,
            citation_audit=citation_audit,
            allowed_chunks=verified_chunks
        )
        span_gen.finish(
            generation_state=post_decision.state,
            citation_precision=citation_audit.citation_precision,
            grounding_ratio=citation_audit.grounding_ratio,
            citation_entailment_precision=citation_audit.citation_entailment_precision
        )

        trace_data = tracer.finish_trace()
        self.metrics_collector.record_query(
            total_latency_ms=trace_data["total_duration_ms"],
            is_blocked=False,
            citation_precision=citation_audit.citation_precision,
            citation_recall=citation_audit.citation_recall,
            grounding_ratio=citation_audit.grounding_ratio
        )

        return PipelineExecutionResult(
            query=query_text,
            generation_state=post_decision.state,
            answer_text=post_decision.final_output,
            confidence=post_decision.confidence,
            citations=citation_audit,
            verified_chunks=verified_chunks,
            quarantined_chunks=quarantined_chunks,
            cov_result=cov_res,
            ring_telemetry={
                "ring_0": q_report,
                "ring_1_drs": drs_report.__dict__,
                "ring_2_risk": risk_report,
                "ring_3_gwcc": consensus_decision.__dict__ if consensus_decision else None
            },
            trace=trace_data
        )
