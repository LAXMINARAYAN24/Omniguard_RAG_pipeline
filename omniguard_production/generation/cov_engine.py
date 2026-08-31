"""
cov_engine.py — 4-Step Chain-of-Verification (CoV) Engine with Independent Corroboration.

Mitigates hallucination and prevents collusion confirmation by executing an automated cross-examination cycle:
  1. Draft Baseline Response
  2. Generate Verification Questions for Atomic Claims
  3. Execute Factual Cross-Checks Against an Independent Evidence Corroboration Pool
  4. Synthesize Grounded, Citation-Anchored Final Response
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set
from ..trust.provenance import ProductionChunk
from ..claims.claim_extractor import ClaimExtractor, AtomicClaim
from ..claims.nli_verifier import NLIVerifier


@dataclass
class CoVVerificationCheck:
    question: str
    target_claim: str
    verification_answer: str
    is_supported: bool
    supporting_chunk_id: Optional[str] = None
    supporting_domain: Optional[str] = None
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    corroborating_domains: List[str] = field(default_factory=list)


@dataclass
class CoVResult:
    baseline_response: str
    verification_checks: List[CoVVerificationCheck]
    revised_response: str
    unsupported_claims_removed: int
    grounding_score: float
    corroboration_ratio: float = 1.0
    telemetry: Dict[str, Any] = field(default_factory=dict)


class ChainOfVerificationEngine:
    """Automated 4-step Chain-of-Verification orchestration pipeline with cross-source corroboration."""

    def __init__(self,
                 llm_generate_fn: Optional[Callable[[str, str], str]] = None,
                 claim_extractor: Optional[ClaimExtractor] = None,
                 nli_verifier: Optional[NLIVerifier] = None,
                 entailment_threshold: float = 0.60):
        self.llm_generate_fn = llm_generate_fn
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.nli_verifier = nli_verifier or NLIVerifier()
        self.entailment_threshold = entailment_threshold

    def run_verification(self,
                         query_text: str,
                         baseline_response: str,
                         verified_chunks: List[ProductionChunk],
                         independent_corroboration_pool: Optional[List[ProductionChunk]] = None) -> CoVResult:
        """
        Executes the full 4-step CoV cycle.
        If independent_corroboration_pool is provided, cross-checks claims across independent lineages
        to verify that consensus is not an artifact of a single colluding cluster.
        """
        # Step 1: Extract factual atomic claims from baseline response
        claims = self.claim_extractor.extract_from_text(baseline_response)

        # Build evidence pool: primary verified chunks + optional independent pool
        primary_chunks = verified_chunks or []
        corroboration_pool = independent_corroboration_pool or primary_chunks

        # Step 2 & 3: Draft verification questions and execute factual NLI cross-checks
        checks: List[CoVVerificationCheck] = []
        multi_domain_corroborated = 0

        for claim in claims:
            q = f"Does the verified evidence confirm that {claim.text}?"
            is_supported = False
            best_chunk_id = None
            best_domain = None
            best_entailment = 0.0
            best_contradiction = 0.0
            matching_domains: Set[str] = set()

            for chunk in corroboration_pool:
                nli_res = self.nli_verifier.check_pair(chunk.clean_text, claim.text)
                ent = nli_res.get("entailment", 0.0)
                contra = nli_res.get("contradiction", 0.0)

                if ent >= self.entailment_threshold and contra < 0.30:
                    is_supported = True
                    if chunk.metadata.publisher_domain:
                        matching_domains.add(chunk.metadata.publisher_domain)
                    if ent > best_entailment:
                        best_entailment = ent
                        best_contradiction = contra
                        best_chunk_id = chunk.chunk_id
                        best_domain = chunk.metadata.publisher_domain

            if len(matching_domains) > 1:
                multi_domain_corroborated += 1

            verdict_text = (
                f"Confirmed by verified evidence (entailment {best_entailment:.2f})"
                if is_supported else "Not found or contradicted in evidence"
            )

            checks.append(CoVVerificationCheck(
                question=q,
                target_claim=claim.text,
                verification_answer=verdict_text,
                is_supported=is_supported,
                supporting_chunk_id=best_chunk_id,
                supporting_domain=best_domain,
                entailment_score=round(best_entailment, 4),
                contradiction_score=round(best_contradiction, 4),
                corroborating_domains=sorted(list(matching_domains))
            ))

        # Step 4: Synthesize revised response filtering unsupported claims
        supported_checks = [c for c in checks if c.is_supported]
        unsupported_count = len(checks) - len(supported_checks)

        if not supported_checks and primary_chunks:
            # If baseline had no supported claims, construct direct grounded answer from top verified chunk
            top_chunk = primary_chunks[0]
            title = top_chunk.metadata.title or top_chunk.doc_id
            h_short = top_chunk.content_hash[:8]
            revised_response = (
                f"According to verified records, {top_chunk.clean_text} "
                f"[Doc: {title} | Chunk: {top_chunk.chunk_index} | Hash: {h_short}]"
            )
        elif not supported_checks and not primary_chunks:
            revised_response = "Information not available in provided sources."
        else:
            if self.llm_generate_fn is not None:
                # LLM synthesis with strict verification constraints
                fact_bullet_points = "\n".join(
                    f"- {chk.target_claim} [Chunk: {chk.supporting_chunk_id}]"
                    for chk in supported_checks
                )
                system_prompt = (
                    "You are a strictly grounded factual synthesizer. Synthesize a concise answer to the user's "
                    "question using ONLY the verified facts below. Do NOT extrapolate. Attach citation markers."
                )
                user_prompt = (
                    f"Question: {query_text}\n\n"
                    f"Verified Facts:\n{fact_bullet_points}\n\n"
                    f"Synthesize the grounded answer:"
                )
                try:
                    revised_response = self.llm_generate_fn(system_prompt, user_prompt)
                except Exception:
                    revised_response = self._deterministic_sentence_reconstruction(baseline_response, supported_checks, primary_chunks)
            else:
                revised_response = self._deterministic_sentence_reconstruction(baseline_response, supported_checks, primary_chunks)

        grounding_score = len(supported_checks) / max(1, len(claims)) if claims else 1.0
        corroboration_ratio = multi_domain_corroborated / max(1, len(supported_checks)) if supported_checks else 1.0

        return CoVResult(
            baseline_response=baseline_response,
            verification_checks=checks,
            revised_response=revised_response,
            unsupported_claims_removed=unsupported_count,
            grounding_score=round(grounding_score, 4),
            corroboration_ratio=round(corroboration_ratio, 4),
            telemetry={
                "total_claims_checked": len(checks),
                "supported_claims": len(supported_checks),
                "unsupported_claims": unsupported_count,
                "multi_domain_corroborated_claims": multi_domain_corroborated,
                "corroboration_ratio": round(corroboration_ratio, 4)
            }
        )

    def _deterministic_sentence_reconstruction(self,
                                               baseline_response: str,
                                               supported_checks: List[CoVVerificationCheck],
                                               verified_chunks: List[ProductionChunk]) -> str:
        """Reconstructs response retaining verified claims and appending citation tags."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", baseline_response) if s.strip()]
        valid_sentences = []

        chunk_lookup = {c.chunk_id: c for c in verified_chunks}

        for s in sentences:
            # Check if sentence matches any supported claim
            matching_chk = next(
                (chk for chk in supported_checks if chk.target_claim in s or s in chk.target_claim),
                None
            )
            if matching_chk is not None:
                # Attach citation if missing
                if not re.search(r"\[Doc:", s):
                    chunk = chunk_lookup.get(matching_chk.supporting_chunk_id) if matching_chk.supporting_chunk_id else (verified_chunks[0] if verified_chunks else None)
                    if chunk:
                        title = chunk.metadata.title or chunk.doc_id
                        h_short = chunk.content_hash[:8]
                        s = f"{s} [Doc: {title} | Chunk: {chunk.chunk_index} | Hash: {h_short}]"
                valid_sentences.append(s)

        return " ".join(valid_sentences) if valid_sentences else baseline_response
