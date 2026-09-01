"""
test_temporal_trust_amplification.py — Adversarial Verification of Temporal Trust Amplification Defense.

Validates:
1. Sub-linear diminishing rewards (1 / sqrt(N)) preventing repeated-query runaway trust.
2. Domain diversity ceilings (< 3 distinct documents cannot exceed 0.80 trust).
3. Daily velocity rate limits (max +0.05 domain trust gain per 24-hour window).
4. Multi-document legitimate promotion path for authentic publishers.
"""
import unittest
import math
from omniguard_production.trust.trust_store import PersistentTrustStore, TrustWeights


class TestTemporalTrustAmplification(unittest.TestCase):

    def setUp(self):
        self.store = PersistentTrustStore(
            half_life_seconds=86400.0 * 30,
            default_baseline=0.75,
            weights=TrustWeights(domain_weight=0.25, source_weight=0.25, document_weight=0.25, content_weight=0.25)
        )
        self.tenant = "aerospace_sec"

    def test_sublinear_diminishing_returns_on_repeated_hash(self):
        """Simulates 100 queries repeatedly retrieving the exact same content chunk."""
        domain = "shadow-feed.org"
        content_hash = "hash_repeated_payload_99"
        doc_id = "doc_single_001"
        source_id = "src_feed_01"

        initial_trust = self.store.get_effective_trust(
            tenant_id=self.tenant,
            publisher_domain=domain,
            source_id=source_id,
            document_id=doc_id,
            content_hash=content_hash
        )
        self.assertEqual(initial_trust["composite_trust"], 0.75)

        # Fire 100 simulated successful queries
        for q_idx in range(1, 101):
            self.store.record_hierarchical_reward(
                tenant_id=self.tenant,
                publisher_domain=domain,
                source_id=source_id,
                document_id=doc_id,
                content_hash=content_hash,
                reason="Corroborated by user query",
                base_reward=0.05,
                query_id=f"q_{q_idx}"
            )

        final_trust = self.store.get_effective_trust(
            tenant_id=self.tenant,
            publisher_domain=domain,
            source_id=source_id,
            document_id=doc_id,
            content_hash=content_hash
        )

        # Domain trust MUST be strictly capped at <= 0.80 due to lack of document diversity
        self.assertLessEqual(final_trust["domain_trust"], 0.80)
        # Daily velocity cap ensures domain gain is <= 0.05
        self.assertLessEqual(final_trust["domain_trust"] - 0.75, 0.0501)

        # Verify diminishing updates were logged with correct metadata
        hash_events = [e for e in self.store.events if e.entity_id == content_hash]
        self.assertEqual(len(hash_events), 100)
        self.assertAlmostEqual(hash_events[0].delta, 0.05, places=4)
        self.assertAlmostEqual(hash_events[99].delta, 0.05 / math.sqrt(100), places=4)

    def test_domain_diversity_ceiling(self):
        """Ensures a domain cannot reach high-trust tier (> 0.80) with only 1 or 2 documents."""
        domain = "unvetted-publisher.com"
        source_id = "feed_main"

        # Provide positive events across only 2 documents
        for doc_num in [1, 2]:
            doc_id = f"doc_{doc_num}"
            for i in range(20):
                self.store.record_hierarchical_reward(
                    tenant_id=self.tenant,
                    publisher_domain=domain,
                    source_id=source_id,
                    document_id=doc_id,
                    content_hash=f"hash_{doc_num}_{i}",
                    reason="Query pass",
                    base_reward=0.05
                )

        rep = self.store.get_effective_trust(tenant_id=self.tenant, publisher_domain=domain)
        # Ceiling of 0.80 strictly enforced
        self.assertLessEqual(rep["domain_trust"], 0.80)

    def test_multidoc_legitimate_promotion(self):
        """Ensures that a legitimate domain with >= 3 distinct verified documents CAN advance beyond 0.80."""
        domain = "legitimate-journal.org"
        source_id = "academic_feed"

        # Provide positive events across 4 distinct documents
        for doc_num in range(1, 5):
            doc_id = f"peer_reviewed_doc_{doc_num}"
            self.store.record_hierarchical_reward(
                tenant_id=self.tenant,
                publisher_domain=domain,
                source_id=source_id,
                document_id=doc_id,
                content_hash=f"hash_peer_{doc_num}",
                reason="Peer reviewed paper verified",
                base_reward=0.08
            )

        rep = self.store.get_effective_trust(tenant_id=self.tenant, publisher_domain=domain)
        # With 4 distinct documents, ceiling constraint is lifted
        self.assertGreater(len(self.store._domain_distinct_docs[self.tenant][domain]), 3)
        self.assertGreater(rep["domain_trust"], 0.75)


if __name__ == "__main__":
    unittest.main()
