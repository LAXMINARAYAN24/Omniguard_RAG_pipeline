"""
test_production_mechanism_live.py — Master Mechanism-Level Counterfactual Verification Suite

Verifies that each defense mechanism in the OmniGuard Production Control Plane
participates in live inference and causally alters pipeline routing and outputs:
1. Ring 1 Spectral SVD DRS: Low-variance tail projection & anomaly score calculation
2. Ring 2 Proposition NLI: Atomic claim extraction, entailment discounting, & contradiction density routing
3. Ring 3 Causal LGO Consensus: Lineage independence gating & Leave-Group-Out contradiction delta isolation
4. Dynamic Trust Store: Hierarchical penalty propagation (chunk -> doc -> source -> domain) & ledger persistence
5. Chain-of-Verification (CoV): Corroboration question verification & ungrounded claim revision
6. 5-State Calibrated Abstention Engine: Clean answer, conflicting evidence, insufficient evidence, security block
7. Full Pipeline Counterfactual Multi-Ring Integration Test
"""
import unittest
import sys
import numpy as np
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Production control plane components
from omniguard_production.pipeline import OmniGuardProductionPipeline, PipelineExecutionResult
from omniguard_production.trust.provenance import DocumentMetadata, ProductionChunk, ProductionDocument, DocumentState
from omniguard_production.trust.trust_store import PersistentTrustStore, TrustEventType
from omniguard_production.embeddings.neural_provider import DenseNeuralEmbeddingProvider
from omniguard_production.embeddings.drs_engine import DRSEngine, DRSModel, DRSConfig
from omniguard_production.claims.claim_extractor import ClaimExtractor, AtomicClaim
from omniguard_production.claims.nli_verifier import NLIVerifier
from omniguard_production.claims.risk_scorer import CalibratedRiskRouter, RoutingAction
from omniguard_production.consensus.evidence_graph import EvidenceGraph, EvidenceCluster
from omniguard_production.consensus.lgo_analyzer import LGOConsensusAnalyzer, ConsensusStatus, GWCCDecision
from omniguard_production.generation.cov_engine import ChainOfVerificationEngine, CoVResult
from omniguard_production.generation.citation_tracker import CitationTracker
from omniguard_production.generation.abstention_engine import CalibratedAbstentionEngine, GenerationState


class TestProductionDefenseMechanismsLive(unittest.TestCase):
    """Deep mechanism-level live counterfactual test suite."""

    @classmethod
    def setUpClass(cls):
        cls.embedding_provider = DenseNeuralEmbeddingProvider()

    def test_ring1_svd_drs_low_variance_projection_anomaly(self):
        """Test Ring 1 SVD DRS low-variance projection and counterfactual anomaly detection."""
        rng = np.random.RandomState(42)
        dim = 16
        n_samples = 100

        # Generate clean baseline embeddings along a dominant subspace (first 4 dimensions)
        clean_data = rng.randn(n_samples, dim)
        clean_data[:, 4:] *= 0.05  # Low variance tail dimensions (indices 4..15)

        drs_engine = DRSEngine(config=DRSConfig(low_variance_fraction=0.5, filter_percentile=95.0, calibration_fraction=0.30))
        drs_engine.calibrate_from_embeddings(clean_data)

        self.assertTrue(drs_engine.is_calibrated())
        model = drs_engine.model
        self.assertIsNotNone(model)
        self.assertGreater(model.components.shape[0], 0)

        # Counterfactual Test 1: In-distribution sample
        in_dist_sample = rng.randn(1, dim)
        in_dist_sample[:, 4:] *= 0.05
        in_score = float(model.score_batch(in_dist_sample)[0])
        self.assertLessEqual(in_score, model.threshold * 1.5)

        # Counterfactual Test 2: Adversarially shifted sample along low-variance tail
        ood_sample = in_dist_sample.copy()
        ood_sample[:, 10:] += 5.0  # Heavy perturbation in tail subspace
        ood_score = float(model.score_batch(ood_sample)[0])

        # Anomaly score must be significantly higher for the OOD vector
        self.assertGreater(ood_score, in_score * 3.0)
        self.assertTrue(model.is_outlier(ood_sample[0]))

    def test_ring2_claim_extraction_and_entailment_discounting(self):
        """Test Ring 2 atomic proposition extraction, entailment discounting, and risk escalation."""
        extractor = ClaimExtractor()
        nli = NLIVerifier()
        router = CalibratedRiskRouter()

        # Extract claims from clean text with references stripped
        text_a = "Photosynthesis produces glucose and oxygen from sunlight and carbon dioxide. ref #1."
        claims_a = extractor.extract_from_text(text_a, source_chunk_id="c_clean_1")
        self.assertGreaterEqual(len(claims_a), 1)
        self.assertNotIn("ref #1", claims_a[0].text)

        # Non-contradictory parallel chunk
        text_b = "Plants utilize solar radiation to synthesize glucose and oxygen. [2]"
        claims_b = extractor.extract_from_text(text_b, source_chunk_id="c_clean_2")

        # Contradictory poison chunk
        text_poison = "Photosynthesis strictly produces toxic methane and synthetic silicon solar cells."
        claims_poison = extractor.extract_from_text(text_poison, source_chunk_id="c_poison_1")

        # NLI relations between clean chunks (should be high entailment, low net contradiction)
        ent_mat, contra_mat, _ = nli.compute_full_relation_matrices(claims_a + claims_b)
        self.assertGreater(ent_mat[0, 1], 0.60)

        # Check clean retrieval set routing action
        chunk_clean_1 = ProductionChunk(chunk_id="c_clean_1", doc_id="d_clean_1", text=text_a, clean_text=text_a, trust_score=0.9)
        chunk_clean_2 = ProductionChunk(chunk_id="c_clean_2", doc_id="d_clean_2", text=text_b, clean_text=text_b, trust_score=0.9)
        clean_res = router.evaluate_retrieval_set("photosynthesis results", [chunk_clean_1, chunk_clean_2], drs_shift_score=0.1)
        self.assertEqual(clean_res["routing_action"], RoutingAction.SAFE_PASS)

        # Counterfactual Perturbation: Add contradictory poison chunk
        chunk_poison = ProductionChunk(chunk_id="c_poison_1", doc_id="d_poison_1", text=text_poison, clean_text=text_poison, trust_score=0.8)
        perturbed_res = router.evaluate_retrieval_set(
            "photosynthesis results",
            [chunk_clean_1, chunk_clean_2, chunk_poison],
            drs_shift_score=0.1
        )

        # Contradiction intensity and composite risk must rise, triggering consensus routing
        self.assertGreater(perturbed_res["nli_contradiction_intensity"], clean_res["nli_contradiction_intensity"])
        self.assertEqual(perturbed_res["routing_action"], RoutingAction.TARGETED_CONSENSUS)

    def test_ring3_causal_lgo_consensus_and_lineage_gating(self):
        """Test Ring 3 Lineage Independence Matrix and Causal Leave-Group-Out contradiction delta."""
        graph_builder = EvidenceGraph(embedding_provider=self.embedding_provider)
        analyzer = LGOConsensusAnalyzer(dominance_ratio=1.4)

        # Create 3 independent clean chunks from distinct publishers
        clean_chunks = [
            ProductionChunk(
                chunk_id="c1",
                doc_id="d_nasa",
                text="The orbital period of Europa around Jupiter is 3.55 Earth days.",
                clean_text="The orbital period of Europa around Jupiter is 3.55 Earth days.",
                trust_score=0.95,
                metadata=DocumentMetadata(publisher_domain="nasa.gov", source_id="nasa_europa")
            ),
            ProductionChunk(
                chunk_id="c2",
                doc_id="d_esa",
                text="Europa completes an orbit around Jupiter in exactly 3.55 days in tidal lock.",
                clean_text="Europa completes an orbit around Jupiter in exactly 3.55 days in tidal lock.",
                trust_score=0.92,
                metadata=DocumentMetadata(publisher_domain="esa.int", source_id="esa_europa")
            ),
            ProductionChunk(
                chunk_id="c3",
                doc_id="d_nature",
                text="Astronomical observations confirm Europa orbits Jupiter every 3.55 days.",
                clean_text="Astronomical observations confirm Europa orbits Jupiter every 3.55 days.",
                trust_score=0.90,
                metadata=DocumentMetadata(publisher_domain="nature.com", source_id="nature_paper")
            ),
        ]

        # Create 2 colluding poison chunks from the same shadow domain
        poison_chunks = [
            ProductionChunk(
                chunk_id="p1",
                doc_id="d_shadow_1",
                text="Europa orbits Jupiter every 14.2 days according to revisionist models.",
                clean_text="Europa orbits Jupiter every 14.2 days according to revisionist models.",
                trust_score=0.75,
                metadata=DocumentMetadata(publisher_domain="shadow-astronomy.org", source_id="shadow_1")
            ),
            ProductionChunk(
                chunk_id="p2",
                doc_id="d_shadow_2",
                text="Recent leaks confirm Europa's true orbit is 14.2 days, not 3.55 days.",
                clean_text="Recent leaks confirm Europa's true orbit is 14.2 days, not 3.55 days.",
                trust_score=0.75,
                metadata=DocumentMetadata(publisher_domain="shadow-astronomy.org", source_id="shadow_2")
            ),
        ]

        all_chunks = clean_chunks + poison_chunks
        for c in all_chunks:
            c.embedding = self.embedding_provider.embed_text(c.clean_text)

        # 1. Verify Lineage Independence Discounting
        M = graph_builder.compute_source_independence_matrix(all_chunks)
        # Independent domains (nasa.gov vs esa.int) -> 1.0
        self.assertEqual(M[0, 1], 1.0)
        # Same domain (shadow-astronomy.org p1 vs p2) -> discounted (<= 0.70)
        self.assertLessEqual(M[3, 4], 0.70)

        # 2. Synthetic pairwise contradiction matrix (high contradiction between clean and poison)
        n = len(all_chunks)
        contra_mat = np.zeros((n, n), dtype=np.float64)
        for i in range(3):
            for j in range(3, 5):
                contra_mat[i, j] = 0.95
                contra_mat[j, i] = 0.95

        # 3. Build graph & detect communities
        G = graph_builder.build_graph(all_chunks, contradiction_matrix=contra_mat)
        clusters = graph_builder.detect_communities(G)

        self.assertEqual(len(clusters), 2)
        # Verify clean cluster has higher aggregate weight due to domain diversity
        clean_cluster = next(c for c in clusters if any(ch.chunk_id == "c1" for ch in c.chunks))
        poison_cluster = next(c for c in clusters if any(ch.chunk_id == "p1" for ch in c.chunks))

        self.assertGreater(clean_cluster.domain_diversity, poison_cluster.domain_diversity)
        self.assertGreater(clean_cluster.evidence_weight, poison_cluster.evidence_weight)

        # 4. Analyze Causal Leave-Group-Out (LGO) Decision
        decision: GWCCDecision = analyzer.analyze_consensus(clusters, all_chunks, contradiction_matrix=contra_mat)

        self.assertIn(decision.status, {ConsensusStatus.CONSENSUS_VERIFIED, ConsensusStatus.COLLUSION_DISCARDED})
        self.assertEqual(len(decision.selected_chunks), 3)
        self.assertEqual(len(decision.quarantined_chunks), 2)
        self.assertTrue(all(c.chunk_id in {"p1", "p2"} for c in decision.quarantined_chunks))

        # Removing the poison cluster must produce a positive contradiction reduction delta
        poison_lgo_delta = decision.counterfactual_deltas.get(poison_cluster.cluster_id, 0.0)
        self.assertGreater(poison_lgo_delta, 0.0)

    def test_trust_store_hierarchical_penalty_and_persistence(self):
        """Test Dynamic Trust Store hierarchical penalty propagation and audit ledger."""
        store = PersistentTrustStore()
        tenant = "aerospace"
        domain = "adversary-news.net"

        initial_rep = store.get_effective_trust(tenant_id=tenant, publisher_domain=domain)
        self.assertEqual(initial_rep["composite_trust"], 0.75)

        # Record penalty on a single chunk from this domain
        store.record_hierarchical_penalty(
            tenant_id=tenant,
            publisher_domain=domain,
            source_id="feed_alpha",
            document_id="doc_xyz",
            content_hash="hash_12345678",
            reason="Adversarial collusion detected by Ring 3 LGO",
            base_penalty=-0.40
        )

        self.assertGreaterEqual(len(store.events), 1)

        # Query effective trust across hierarchy
        penalized_domain_rep = store.get_effective_trust(tenant_id=tenant, publisher_domain=domain)
        self.assertLess(penalized_domain_rep["composite_trust"], 0.75)
        self.assertLess(penalized_domain_rep["domain_trust"], 0.75)

        # Clean unrelated domain should remain unaffected
        unrelated_rep = store.get_effective_trust(tenant_id=tenant, publisher_domain="nasa.gov")
        self.assertEqual(unrelated_rep["composite_trust"], 0.75)

    def test_chain_of_verification_and_grounding_audit(self):
        """Test Chain-of-Verification (CoV) claim generation, corroboration, and revision."""
        cov = ChainOfVerificationEngine()
        tracker = CitationTracker()

        corroboration_chunks = [
            ProductionChunk(
                chunk_id="c_fact_1",
                doc_id="d_physics",
                text="The speed of light in vacuum is approximately 299,792 kilometers per second.",
                clean_text="The speed of light in vacuum is approximately 299,792 kilometers per second.",
                trust_score=0.98,
                metadata=DocumentMetadata(title="Physics Handbook", publisher_domain="nist.gov")
            )
        ]

        # Case 1: Grounded response matching corroboration pool
        grounded_response = "The speed of light in a vacuum is 299,792 km/s. [Doc: Physics Handbook | Chunk: 0 | Hash: a1b2c3d4]"
        cov_grounded = cov.run_verification(
            query_text="What is the speed of light?",
            baseline_response=grounded_response,
            verified_chunks=corroboration_chunks,
            independent_corroboration_pool=corroboration_chunks
        )

        self.assertGreaterEqual(cov_grounded.grounding_score, 0.70)
        self.assertEqual(cov_grounded.unsupported_claims_removed, 0)

        # Case 2: Hallucinated response with unsupported claim
        unsupported_response = "The speed of light was proven to be 500,000 km/s by secret laboratory experiments."
        cov_hallucinated = cov.run_verification(
            query_text="What is the speed of light?",
            baseline_response=unsupported_response,
            verified_chunks=corroboration_chunks,
            independent_corroboration_pool=corroboration_chunks
        )

        # Unsupported claims must be flagged and revised
        self.assertLess(cov_hallucinated.grounding_score, 0.50)
        self.assertGreaterEqual(cov_hallucinated.unsupported_claims_removed, 1)

    def test_calibrated_abstention_engine_states(self):
        """Test 5-State Calibrated Abstention Engine state emissions under varied conditions."""
        abstention = CalibratedAbstentionEngine()

        # 1. SECURITY_BLOCK on Ring 0 injection detection
        sec_block = abstention.evaluate_pre_generation(is_query_blocked=True, consensus_decision=None, retrieved_chunks=[])
        self.assertEqual(sec_block.state, GenerationState.SECURITY_BLOCK)
        self.assertFalse(sec_block.can_proceed_to_generate)
        self.assertIn("Request blocked", sec_block.final_output)

        # 2. INSUFFICIENT_EVIDENCE on empty retrieval set
        no_evidence = abstention.evaluate_pre_generation(is_query_blocked=False, consensus_decision=None, retrieved_chunks=[])
        self.assertEqual(no_evidence.state, GenerationState.INSUFFICIENT_EVIDENCE)
        self.assertFalse(no_evidence.can_proceed_to_generate)

        # 3. CONFLICTING_EVIDENCE on unresolvable contradiction
        conflicting_gwcc = GWCCDecision(
            status=ConsensusStatus.CONFLICTING_POOLS,
            selected_chunks=[],
            quarantined_chunks=[],
            confidence_score=0.30,
            explanation="Equal weight opposing pools."
        )
        sample_retrieved = [
            ProductionChunk(chunk_id="c1", doc_id="d1", text="Sample fact A", clean_text="Sample fact A"),
            ProductionChunk(chunk_id="c2", doc_id="d2", text="Sample fact B", clean_text="Sample fact B")
        ]
        conflicting = abstention.evaluate_pre_generation(is_query_blocked=False, consensus_decision=conflicting_gwcc, retrieved_chunks=sample_retrieved)
        self.assertEqual(conflicting.state, GenerationState.CONFLICTING_EVIDENCE)
        self.assertFalse(conflicting.can_proceed_to_generate)
        self.assertIn("conflicting", conflicting.final_output.lower())

    def test_end_to_end_pipeline_live_counterfactual_defense(self):
        """End-to-End Live Integration: Ingest multi-source clean & colluding poison docs, query, and verify defense."""
        pipeline = OmniGuardProductionPipeline(tenant_id="deep_space")

        # Ingest 3 clean authoritative documents
        pipeline.ingest_document(
            raw_text="The Hubble Space Telescope was deployed in April 1990 by Space Shuttle Discovery into low Earth orbit.",
            metadata=DocumentMetadata(title="Hubble Deployment", publisher_domain="nasa.gov", source_id="nasa_hst", tenant_id="deep_space")
        )
        pipeline.ingest_document(
            raw_text="Operating in low Earth orbit since 1990, the Hubble Space Telescope has made over 1.5 million observations.",
            metadata=DocumentMetadata(title="Hubble Operations", publisher_domain="esa.int", source_id="esa_hst", tenant_id="deep_space")
        )
        pipeline.ingest_document(
            raw_text="Hubble Space Telescope deployment occurred in 1990 via the STS-31 mission.",
            metadata=DocumentMetadata(title="STS-31 Mission Report", publisher_domain="smithsonian.org", source_id="smithsonian_hst", tenant_id="deep_space")
        )

        # Ingest 2 colluding poison documents attempting to rewrite history
        pipeline.ingest_document(
            raw_text="The Hubble Space Telescope was secretly launched in 2015 from a clandestine ocean platform ref #1.",
            metadata=DocumentMetadata(title="Secret Hubble Dossier", publisher_domain="shadow-leaks.net", source_id="shadow_doc_1", tenant_id="deep_space")
        )
        pipeline.ingest_document(
            raw_text="Declassified memos confirm Hubble was launched in 2015 rather than 1990 ref #2.",
            metadata=DocumentMetadata(title="Declassified Hubble", publisher_domain="shadow-leaks.net", source_id="shadow_doc_2", tenant_id="deep_space")
        )

        # Execute Live Query
        result: PipelineExecutionResult = pipeline.query("When was the Hubble Space Telescope launched?", tenant_id="deep_space")

        # Verify ground-truth answer generation and quarantine isolation
        self.assertIn(result.generation_state, {GenerationState.ANSWER, GenerationState.PARTIAL_ANSWER})
        self.assertIn("1990", result.answer_text)
        self.assertNotIn("2015", result.answer_text)

        # Verify poison chunks were quarantined by Ring 3 LGO
        self.assertGreaterEqual(len(result.quarantined_chunks), 1)
        self.assertTrue(any(c.metadata.publisher_domain == "shadow-leaks.net" for c in result.quarantined_chunks))

        # Verify citation audit and grounding
        self.assertGreater(len(result.verified_chunks), 0)
        self.assertTrue(all(c.metadata.publisher_domain != "shadow-leaks.net" for c in result.verified_chunks))

    def test_production_pipeline_uses_real_drs_engine(self):
        """Wire-level verification: Ensure DRSEngine is on the critical retrieval path and evaluated during pipeline.query()."""
        pipeline = OmniGuardProductionPipeline(tenant_id="physics")

        # Ingest clean physics documents
        for i in range(8):
            pipeline.ingest_document(
                raw_text=f"The gravitational constant G is approximately 6.67430e-11 m^3 kg^-1 s^-2 according to NIST standard ref #{i+1}.",
                metadata=DocumentMetadata(
                    title=f"NIST Physics Standard {i+1}",
                    publisher_domain="nist.gov",
                    source_id=f"nist_g_{i+1}",
                    tenant_id="physics"
                )
            )

        self.assertTrue(pipeline.drs_engine.is_calibrated(), "DRS Engine must be auto-calibrated after document ingestion")

        # Execute query
        res = pipeline.query("What is the gravitational constant G?", tenant_id="physics")

        # Assert DRS telemetry was captured and executed during the live query
        self.assertIn("ring_1_drs", res.ring_telemetry, "ring_1_drs must be present in query telemetry")
        drs_telemetry = res.ring_telemetry["ring_1_drs"]
        self.assertTrue(drs_telemetry.get("is_calibrated"), "DRS telemetry must indicate calibrated status")
        self.assertIn("max_drs_score", drs_telemetry, "DRS telemetry must contain calculated max_drs_score")
        self.assertGreaterEqual(drs_telemetry["max_drs_score"], 0.0)

    def test_production_pipeline_uses_llm_callback(self):
        """Wire-level verification: Ensure custom LLM generator callback receives verified source prompts."""
        call_log = []

        def sentinel_llm(system_prompt: str, user_prompt: str) -> str:
            call_log.append({
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            })
            return "The gravitational constant is exactly 6.67430e-11 m^3 kg^-1 s^-2. [Doc: NIST Physics Standard 1 | Chunk: 0 | Hash: a1b2c3d4]"

        pipeline = OmniGuardProductionPipeline(
            llm_generator_fn=sentinel_llm,
            tenant_id="physics"
        )

        pipeline.ingest_document(
            raw_text="The gravitational constant G is approximately 6.67430e-11 m^3 kg^-1 s^-2 according to NIST standard ref #1.",
            metadata=DocumentMetadata(
                title="NIST Physics Standard 1",
                publisher_domain="nist.gov",
                source_id="nist_g_1",
                tenant_id="physics"
            )
        )

        res = pipeline.query("What is the value of G?", tenant_id="physics")

        # Verify sentinel LLM was called with the assembled prompt
        self.assertEqual(len(call_log), 1, "sentinel_llm must be invoked exactly once during generation")
        self.assertIn("gravitational constant", call_log[0]["user_prompt"])
        self.assertIn("NIST Physics Standard 1", call_log[0]["user_prompt"])
        self.assertIn("6.67430e-11", res.answer_text)

    def test_lgo_changes_decision(self):
        """Causal verification: Removing the colluding group causally changes consensus and reduces contradiction."""
        graph_builder = EvidenceGraph(embedding_provider=self.embedding_provider)
        analyzer = LGOConsensusAnalyzer(dominance_ratio=1.4)

        # 3 clean chunks
        clean_chunks = [
            ProductionChunk(
                chunk_id="c1", doc_id="d1",
                text="The speed of sound in dry air at 20 C is 343 meters per second.",
                clean_text="The speed of sound in dry air at 20 C is 343 meters per second.",
                trust_score=0.95,
                metadata=DocumentMetadata(publisher_domain="noaa.gov", source_id="s1")
            ),
            ProductionChunk(
                chunk_id="c2", doc_id="d2",
                text="At 20 degrees Celsius, sound travels at 343 m/s through air.",
                clean_text="At 20 degrees Celsius, sound travels at 343 m/s through air.",
                trust_score=0.90,
                metadata=DocumentMetadata(publisher_domain="physics.org", source_id="s2")
            ),
        ]

        # 2 colluding poison chunks
        poison_chunks = [
            ProductionChunk(
                chunk_id="p1", doc_id="d3",
                text="The speed of sound in air is 999 meters per second.",
                clean_text="The speed of sound in air is 999 meters per second.",
                trust_score=0.70,
                metadata=DocumentMetadata(publisher_domain="fake-acoustics.com", source_id="s3")
            ),
            ProductionChunk(
                chunk_id="p2", doc_id="d4",
                text="Sound travels at 999 m/s in air as proven by new experiments.",
                clean_text="Sound travels at 999 m/s in air as proven by new experiments.",
                trust_score=0.70,
                metadata=DocumentMetadata(publisher_domain="fake-acoustics.com", source_id="s4")
            ),
        ]

        all_chunks = clean_chunks + poison_chunks
        for c in all_chunks:
            c.embedding = self.embedding_provider.embed_text(c.clean_text)

        contra_mat = np.zeros((4, 4), dtype=np.float64)
        for i in range(2):
            for j in range(2, 4):
                contra_mat[i, j] = 0.90
                contra_mat[j, i] = 0.90

        # Run on full set (clean + colluding poison)
        G_full = graph_builder.build_graph(all_chunks, contradiction_matrix=contra_mat)
        clusters_full = graph_builder.detect_communities(G_full)
        decision_full = analyzer.analyze_consensus(clusters_full, all_chunks, contradiction_matrix=contra_mat)

        # In full set, poison cluster is quarantined, lgo_delta is positive
        self.assertEqual(len(decision_full.quarantined_chunks), 2)
        poison_cluster = next(c for c in clusters_full if any(ch.chunk_id == "p1" for ch in c.chunks))
        delta = decision_full.counterfactual_deltas.get(poison_cluster.cluster_id, 0.0)
        self.assertGreater(delta, 0.0, "Removing poison cluster must produce a positive contradiction reduction delta")

        # Run on clean set only (counterfactual baseline)
        G_clean = graph_builder.build_graph(clean_chunks, contradiction_matrix=np.zeros((2, 2)))
        clusters_clean = graph_builder.detect_communities(G_clean)
        decision_clean = analyzer.analyze_consensus(clusters_clean, clean_chunks, contradiction_matrix=np.zeros((2, 2)))

        self.assertEqual(len(decision_clean.quarantined_chunks), 0)
        self.assertEqual(len(decision_clean.selected_chunks), 2)
        self.assertNotEqual(decision_full.quarantined_chunks, decision_clean.quarantined_chunks)


if __name__ == "__main__":
    unittest.main()
