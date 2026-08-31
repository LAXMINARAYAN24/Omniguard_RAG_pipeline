"""
rag_defense_engine.py — Core RAG Defense Pipeline & Deep Telemetry Capture

Bridges user queries, attack regimes, and multi-system comparisons with the
underlying unified_rag_defense architecture.
"""
from dataclasses import dataclass, field, asdict
import json
import time
import numpy as np
from numpy.random import default_rng
from typing import Any, Dict, List, Optional, Set, Tuple

from unified_rag_defense.corpus import World, Document, Query
from unified_rag_defense.bench_common import fresh_docs, build_world
from unified_rag_defense.topics_data import TOPICS

TOPIC_MAP = {i: t for i, t in enumerate(TOPICS)}
from unified_rag_defense.query_guard import screen_query, effective_embedding, _repetition_ratio, REPETITION_THRESHOLD
from unified_rag_defense.drs_filter import DRSFilter
from unified_rag_defense.retrieval import top_k
from unified_rag_defense.risk_router import route, cohesion, answer_contention, RISK_THRESHOLD, CONTENTION_THRESHOLD
from unified_rag_defense.gwcc_consensus import gwcc_consensus, weighted_majority
from unified_rag_defense.omniguard_pipeline import DynamicTrustStore, OmniGuardResult
from unified_rag_defense.attack_simulator import (
    run_attack, apply_standard_poison, apply_pidp_compound,
    apply_collusion, apply_collusion_stealth, apply_silent_retrieval, WRONG_ANSWER_TAG
)
from unified_rag_defense.baselines import (
    vanilla_rag, drs_only, shieldrag_only, raguard_zkip, trishield, _corpus_centroid
)


@dataclass
class Ring0Telemetry:
    flagged: bool
    repetition_ratio: float
    threshold: float
    action_taken: str
    removed_suffix: Optional[str]
    sanitized_text: str
    reason: str


@dataclass
class DroppedDocInfo:
    doc_id: str
    anomaly_score: float
    is_poison: bool
    snippet: str
    claim_answer: str


@dataclass
class Ring1Telemetry:
    total_candidate_docs: int
    kept_docs_count: int
    dropped_docs_count: int
    drs_threshold: float
    dropped_documents: List[Dict[str, Any]]


@dataclass
class RetrievedDocInfo:
    rank: int
    doc_id: str
    cosine_similarity: float
    trust_score: float
    effective_weight: float
    claim_answer: str
    is_poison: bool
    text_snippet: str
    label: str


@dataclass
class Ring2Telemetry:
    embedding_cohesion: float
    cohesion_threshold: float
    cohesion_delta: float
    answer_contention: float
    contention_threshold: float
    route_decision: str
    escalation_reason: str


@dataclass
class Ring3Telemetry:
    invoked: bool
    full_set_answer: Optional[str]
    leave_one_out_flips: List[str]
    pairwise_cliques_implicated: List[str]
    excluded_doc_ids: List[str]
    consensus_answer: Optional[str]
    additional_calls: int


@dataclass
class DefenseTelemetry:
    ring0: Ring0Telemetry
    ring1: Ring1Telemetry
    retrieval: List[RetrievedDocInfo]
    ring2: Ring2Telemetry
    ring3: Ring3Telemetry
    trust_store_updates: List[Dict[str, Any]]


@dataclass
class SystemExecutionResult:
    system_name: str
    answer: Optional[str]
    calls: int
    route: str
    is_correct: bool
    is_attack_success: bool
    latency_ms: float
    telemetry: Optional[DefenseTelemetry] = None
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)


class RAGDefenseEngine:
    """
    Stateful engine maintaining the World corpus, DRS filter, and Dynamic Trust Store.
    Provides deep telemetry instrumentation across all 4 defense rings.
    """
    def __init__(self, seed: int = 42, docs_per_topic: int = 30):
        self.seed = seed
        self.rng = default_rng(seed)
        # Build clean World (480 clean documents across 16 topics), DRS Filter, and corpus centroid
        self.world, self.drs, self.centroid = build_world(seed=seed, docs_per_topic=docs_per_topic, embedding_space="tfidf")
        # Initialize Dynamic Trust Store
        self.trust_store = DynamicTrustStore()
        # Custom Injected User Poison Pool
        self.custom_poison_docs: List[Document] = []
        self.custom_doc_counter = 0

    def reset_trust_store(self):
        """Reset all document trust scores to 1.0."""
        self.trust_store = DynamicTrustStore()
        self.custom_poison_docs = []

    def inject_custom_poison(self, topic_id: int, text: str, target_answer: str) -> Document:
        """Add a custom user-crafted adversarial document into the live corpus."""
        self.custom_doc_counter += 1
        doc_id = f"custom_poison_{self.custom_doc_counter}"
        emb = self.world.embed(text)
        doc = Document(
            doc_id=doc_id,
            text=text,
            embedding=emb,
            topic_id=topic_id,
            label="wrong",
            answer=target_answer,
            is_poison=True,
            attack_type="custom",
            trust_score=1.0
        )
        self.custom_poison_docs.append(doc)
        return doc

    def match_topic_from_query(self, query_text: str) -> Tuple[int, Dict[str, Any]]:
        """Identify closest matching topic for arbitrary user input."""
        q_lower = query_text.lower()
        # Direct keyword / topic name matching
        for i, t in enumerate(TOPICS):
            t_name = t["name"].lower().replace("_", " ")
            if t_name in q_lower or t["name"].lower() in q_lower:
                return i, t
            for kw in t["keywords"]:
                if kw.lower() in q_lower:
                    return i, t

        # Default fallback to topic 0
        return 0, TOPICS[0]

    def create_query_object(self, query_text: str, topic_id: Optional[int] = None,
                            adversarial_suffix: Optional[str] = None) -> Query:
        """Construct a Query dataclass instance with embedding."""
        if topic_id is None:
            t_id, topic_data = self.match_topic_from_query(query_text)
        else:
            t_id = topic_id
            topic_data = TOPIC_MAP.get(t_id, TOPICS[0])

        correct_ans = topic_data["answer"]
        base_emb = self.world.embed(query_text)

        suffix_vec = None
        if adversarial_suffix:
            # Suffix vector in TF-IDF space
            suffix_vec = self.world.vectorizer.transform([adversarial_suffix]).toarray()[0]
            if np.linalg.norm(suffix_vec) > 0:
                suffix_vec = suffix_vec / np.linalg.norm(suffix_vec)

        return Query(
            query_id=f"q_{int(time.time() * 1000) % 100000}",
            topic_id=t_id,
            text=query_text,
            base_embedding=base_emb,
            correct_answer=correct_ans,
            suffix_text=adversarial_suffix,
            suffix_vector=suffix_vec
        )

    def prepare_attack_pool(self, query: Query, attack_type: str,
                            k_poison: int = 3, target_answer: Optional[str] = None) -> Tuple[Query, List[Document]]:
        """
        Generate candidate document pool and apply selected attack regime.
        """
        pool = fresh_docs(self.world)

        # Include any custom user-injected poison docs
        if self.custom_poison_docs:
            pool.extend(self.custom_poison_docs)

        active_query = query
        if attack_type == "clean":
            return active_query, pool

        elif attack_type == "standard":
            # Keyword stuffing attack
            active_query, poisons = apply_standard_poison(self.world, query, self.rng)
            pool.extend(poisons)

        elif attack_type == "pidp":
            # Prompt Injection Data Poisoning with adversarial suffix
            active_query, poisons = apply_pidp_compound(self.world, query, self.rng)
            pool.extend(poisons)

        elif attack_type == "collusion" or attack_type == "collusion_minor" or attack_type == "collusion_major":
            k = k_poison if attack_type == "collusion" else (2 if attack_type == "collusion_minor" else 3)
            active_query, poisons = apply_collusion(self.world, query, self.rng, k_poison=k)
            pool.extend(poisons)

        elif attack_type == "collusion_stealth":
            # CATG clean-template stealth collusion attack
            active_query, poisons = apply_collusion_stealth(self.world, query, self.rng, k_poison=k_poison)
            pool.extend(poisons)

        elif attack_type == "silent":
            # Subspace-aligned silent injection
            active_query, poisons = apply_silent_retrieval(self.world, query, self.rng)
            pool.extend(poisons)

        return active_query, pool

    def run_omniguard_with_telemetry(self, query: Query, candidate_docs: List[Document],
                                     persist_trust: bool = True) -> SystemExecutionResult:
        """
        Executes OmniGuard-RAG with deep telemetry capture across Rings 0, 1, 2, and 3.
        """
        t0 = time.time()
        # --- Ring 0: Query Guard ---
        guard = screen_query(query)
        sanitized_query = guard.sanitized_query
        rep_ratio = _repetition_ratio(query.suffix_text) if query.suffix_text else 0.0

        ring0_tel = Ring0Telemetry(
            flagged=guard.flagged,
            repetition_ratio=round(float(rep_ratio), 3),
            threshold=REPETITION_THRESHOLD,
            action_taken="stripped_suffix" if guard.flagged else "pass_through",
            removed_suffix=query.suffix_text if guard.flagged else None,
            sanitized_text=sanitized_query.text,
            reason=guard.reason
        )

        # --- Dynamic Trust Store Application ---
        self.trust_store.apply(candidate_docs)

        # --- Ring 1: Spectral Guard (DRS Filter) ---
        total_candidates = len(candidate_docs)
        filt = self.drs.filter(candidate_docs)
        dropped_info = []
        for doc in filt.dropped:
            score = self.drs.score(doc)
            snippet = doc.text[:120] + "..." if len(doc.text) > 120 else doc.text
            dropped_info.append({
                "doc_id": doc.doc_id,
                "anomaly_score": round(float(score), 3),
                "is_poison": doc.is_poison,
                "snippet": snippet,
                "claim_answer": doc.answer
            })

        ring1_tel = Ring1Telemetry(
            total_candidate_docs=total_candidates,
            kept_docs_count=len(filt.kept),
            dropped_docs_count=len(filt.dropped),
            drs_threshold=round(float(self.drs.threshold), 3),
            dropped_documents=dropped_info
        )

        # --- Retrieval ---
        q_emb = effective_embedding(sanitized_query, self.world)
        entries = top_k(q_emb, filt.kept, k=5)
        top_docs = [d for d, _ in entries]

        retrieval_tel = []
        for rank, (doc, sim) in enumerate(entries, start=1):
            snippet = doc.text[:140] + "..." if len(doc.text) > 140 else doc.text
            retrieval_tel.append(RetrievedDocInfo(
                rank=rank,
                doc_id=doc.doc_id,
                cosine_similarity=round(float(sim), 3),
                trust_score=round(float(doc.trust_score), 3),
                effective_weight=round(float(sim * doc.trust_score), 3),
                claim_answer=doc.answer,
                is_poison=doc.is_poison,
                text_snippet=snippet,
                label=doc.label
            ))

        if not entries:
            elapsed_ms = (time.time() - t0) * 1000.0
            return SystemExecutionResult(
                system_name="OmniGuard-RAG",
                answer=None,
                calls=1,
                route="fast",
                is_correct=False,
                is_attack_success=False,
                latency_ms=elapsed_ms,
                retrieved_docs=[asdict(r) for r in retrieval_tel]
            )

        # --- Ring 2: Risk-Aware Router ---
        coh = cohesion(top_docs)
        coh_delta = 1.0 - coh
        cont = answer_contention(entries)
        decision = route(entries)

        escalation_reason = (
            f"Escalated to Deep GWCC (Cohesion={coh:.3f} < {RISK_THRESHOLD} or Contention={cont:.3f} >= {CONTENTION_THRESHOLD})"
            if decision.route == "deep"
            else f"Low Risk (Cohesion={coh:.3f} >= {RISK_THRESHOLD} and Contention={cont:.3f} < {CONTENTION_THRESHOLD})"
        )

        ring2_tel = Ring2Telemetry(
            embedding_cohesion=round(float(coh), 3),
            cohesion_threshold=RISK_THRESHOLD,
            cohesion_delta=round(float(coh_delta), 3),
            answer_contention=round(float(cont), 3),
            contention_threshold=CONTENTION_THRESHOLD,
            route_decision=decision.route,
            escalation_reason=escalation_reason
        )

        # --- Execution Path ---
        trust_updates = []
        if decision.route == "fast":
            final_answer = weighted_majority(entries)
            calls = 1
            ring3_tel = Ring3Telemetry(
                invoked=False,
                full_set_answer=final_answer,
                leave_one_out_flips=[],
                pairwise_cliques_implicated=[],
                excluded_doc_ids=[],
                consensus_answer=final_answer,
                additional_calls=0
            )
            if persist_trust:
                prev_scores = {d.doc_id: d.trust_score for d in top_docs}
                self.trust_store.update(top_docs, final_answer, implicated=set())
                for d in top_docs:
                    new_score = self.trust_store.scores.get(d.doc_id, 1.0)
                    if new_score != prev_scores[d.doc_id]:
                        trust_updates.append({
                            "doc_id": d.doc_id,
                            "previous": round(prev_scores[d.doc_id], 3),
                            "new": round(new_score, 3),
                            "change": "boosted"
                        })
        else:
            # Deep GWCC Path
            gwcc = gwcc_consensus(entries, self.rng)
            final_answer = gwcc.answer
            calls = 1 + gwcc.calls
            implicated_ids = list(gwcc.implicated_doc_ids)

            ring3_tel = Ring3Telemetry(
                invoked=True,
                full_set_answer=weighted_majority(entries),
                leave_one_out_flips=gwcc.loo_implicated_ids,
                pairwise_cliques_implicated=gwcc.pair_implicated_ids,
                excluded_doc_ids=implicated_ids,
                consensus_answer=final_answer,
                additional_calls=gwcc.calls
            )

            if persist_trust:
                prev_scores = {d.doc_id: d.trust_score for d in top_docs}
                implicated_set = gwcc.implicated_doc_ids
                self.trust_store.update(top_docs, gwcc.answer, implicated=implicated_set)
                for d in top_docs:
                    new_score = self.trust_store.scores.get(d.doc_id, 1.0)
                    if new_score != prev_scores[d.doc_id]:
                        trust_updates.append({
                            "doc_id": d.doc_id,
                            "previous": round(prev_scores[d.doc_id], 3),
                            "new": round(new_score, 3),
                            "change": "penalized" if d.doc_id in implicated_set else "boosted"
                        })

        elapsed_ms = (time.time() - t0) * 1000.0
        is_correct = (final_answer == query.correct_answer)
        is_attack_success = (final_answer == WRONG_ANSWER_TAG)

        telemetry = DefenseTelemetry(
            ring0=ring0_tel,
            ring1=ring1_tel,
            retrieval=retrieval_tel,
            ring2=ring2_tel,
            ring3=ring3_tel,
            trust_store_updates=trust_updates
        )

        return SystemExecutionResult(
            system_name="OmniGuard-RAG",
            answer=final_answer,
            calls=calls,
            route=decision.route,
            is_correct=is_correct,
            is_attack_success=is_attack_success,
            latency_ms=round(elapsed_ms, 2),
            telemetry=telemetry,
            retrieved_docs=[asdict(r) for r in retrieval_tel]
        )

    def run_side_by_side_comparison(self, query: Query, candidate_docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Evaluates the exact same query, attack, and candidate document pool across
        all 6 state-of-the-art systems simultaneously.
        """
        results = []

        # 1. Vanilla RAG
        t0 = time.time()
        res_vanilla = vanilla_rag(query, candidate_docs, self.world)
        results.append({
            "system_id": "vanilla_rag",
            "name": "Vanilla RAG (No Defense)",
            "answer": res_vanilla.answer,
            "calls": res_vanilla.calls,
            "is_correct": res_vanilla.answer == query.correct_answer,
            "is_attack_success": res_vanilla.answer == WRONG_ANSWER_TAG,
            "latency_ms": round((time.time() - t0) * 1000.0, 2),
            "defense_action": "None (Direct top-k vote)"
        })

        # 2. DRS Only
        t0 = time.time()
        res_drs = drs_only(query, candidate_docs, self.drs, self.world)
        results.append({
            "system_id": "drs_only",
            "name": "DRS Spectral Filter Only",
            "answer": res_drs.answer,
            "calls": res_drs.calls,
            "is_correct": res_drs.answer == query.correct_answer,
            "is_attack_success": res_drs.answer == WRONG_ANSWER_TAG,
            "latency_ms": round((time.time() - t0) * 1000.0, 2),
            "defense_action": "Spectral low-variance SVD filter"
        })

        # 3. ShieldRAG Only
        t0 = time.time()
        res_shield = shieldrag_only(query, candidate_docs, self.world)
        results.append({
            "system_id": "shieldrag",
            "name": "ShieldRAG (Iterative Reweighting)",
            "answer": res_shield.answer,
            "calls": res_shield.calls,
            "is_correct": res_shield.answer == query.correct_answer,
            "is_attack_success": res_shield.answer == WRONG_ANSWER_TAG,
            "latency_ms": round((time.time() - t0) * 1000.0, 2),
            "defense_action": "4x iterative score reweighting"
        })

        # 4. RAGuard / ZKIP
        t0 = time.time()
        res_raguard = raguard_zkip(query, candidate_docs, self.world)
        results.append({
            "system_id": "raguard",
            "name": "RAGuard / ZKIP",
            "answer": res_raguard.answer,
            "calls": res_raguard.calls,
            "is_correct": res_raguard.answer == query.correct_answer,
            "is_attack_success": res_raguard.answer == WRONG_ANSWER_TAG,
            "latency_ms": round((time.time() - t0) * 1000.0, 2),
            "defense_action": "Singleton leave-one-out stability check"
        })

        # 5. TriShield
        t0 = time.time()
        res_trishield = trishield(query, candidate_docs, self.world, self.centroid)
        results.append({
            "system_id": "trishield",
            "name": "TriShieldRAG",
            "answer": res_trishield.answer,
            "calls": res_trishield.calls,
            "is_correct": res_trishield.answer == query.correct_answer,
            "is_attack_success": res_trishield.answer == WRONG_ANSWER_TAG,
            "latency_ms": round((time.time() - t0) * 1000.0, 2),
            "defense_action": "Perplexity + Pattern + Outlier distance"
        })

        # 6. OmniGuard-RAG
        omni_exec = self.run_omniguard_with_telemetry(query, candidate_docs, persist_trust=False)
        results.append({
            "system_id": "omniguard",
            "name": "OmniGuard-RAG (4-Ring Defense)",
            "answer": omni_exec.answer,
            "calls": omni_exec.calls,
            "is_correct": omni_exec.is_correct,
            "is_attack_success": omni_exec.is_attack_success,
            "latency_ms": omni_exec.latency_ms,
            "defense_action": f"4-Ring ({omni_exec.route.upper()} path: Ring0->1->2{'->3' if omni_exec.route=='deep' else ''})"
        })

        return results


# Global singleton instance
GLOBAL_RAG_ENGINE = RAGDefenseEngine()
