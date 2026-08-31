"""
cov_engine.py — 4-Step Chain-of-Verification (CoV) Engine.

Mitigates hallucination by executing an automated cross-examination cycle:
  1. Draft Baseline Response
  2. Generate Verification Questions
  3. Execute Factual Cross-Checks
  4. Synthesize Grounded, Fact-Checked Final Response
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
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


@dataclass
class CoVResult:
    baseline_response: str
    verification_checks: List[CoVVerificationCheck]
    revised_response: str
    unsupported_claims_removed: int
    grounding_score: float
    telemetry: Dict[str, Any] = field(default_factory=dict)


class ChainOfVerificationEngine:
    """Automated 4-step Chain-of-Verification orchestration pipeline."""

    def __init__(self,
                 llm_generate_fn: Optional[Callable[[str, str], str]] = None,
                 claim_extractor: Optional[ClaimExtractor] = None,
                 nli_verifier: Optional[NLIVerifier] = None):
        self.llm_generate_fn = llm_generate_fn
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.nli_verifier = nli_verifier or NLIVerifier()

    def run_verification(self,
                         query_text: str,
                         baseline_response: str,
                         verified_chunks: List[ProductionChunk]) -> CoVResult:
        """Executes the full 4-step CoV cycle."""
        # Step 1: Extract factual claims from baseline response
        claims = self.claim_extractor.extract_from_text(baseline_response)

        # Step 2: Draft verification questions for each extracted claim
        checks: List[CoVVerificationCheck] = []
        for claim in claims:
            q = f"Does the verified evidence confirm that {claim.text}?"
            # Step 3: Execute factual verification against grounded chunks
            is_supported = False
            best_chunk_id = None
            best_entailment = 0.0

            for chunk in verified_chunks:
                nli_res = self.nli_verifier.check_pair(chunk.clean_text, claim.text)
                if nli_res.get("entailment", 0.0) >= 0.65 and nli_res.get("contradiction", 0.0) < 0.30:
                    is_supported = True
                    best_chunk_id = chunk.chunk_id
                    best_entailment = nli_res.get("entailment", 0.0)
                    break

            checks.append(CoVVerificationCheck(
                question=q,
                target_claim=claim.text,
                verification_answer="Confirmed by verified evidence" if is_supported else "Not found in evidence",
                is_supported=is_supported,
                supporting_chunk_id=best_chunk_id
            ))

        # Step 4: Synthesize revised response filtering unsupported claims
        supported_checks = [c for c in checks if c.is_supported]
        unsupported_count = len(checks) - len(supported_checks)

        if not supported_checks and verified_chunks:
            # If baseline had no supported claims, construct direct grounded answer from top chunk
            top_chunk = verified_chunks[0]
            title = top_chunk.metadata.title or top_chunk.doc_id
            h_short = top_chunk.content_hash[:8]
            revised_response = (
                f"According to verified records, {top_chunk.clean_text} "
                f"[Doc: {title} | Chunk: {top_chunk.chunk_index} | Hash: {h_short}]"
            )
        elif not supported_checks and not verified_chunks:
            revised_response = "Information not available in provided sources."
        else:
            # Reconstruct response retaining verified claims and appending citation tags
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", baseline_response) if s.strip()]
            valid_sentences = []
            for s in sentences:
                # Check if sentence matches any supported claim
                if any(chk.target_claim in s or s in chk.target_claim for chk in supported_checks):
                    # Attach citation if missing
                    if not re.search(r"\[Doc:", s):
                        matching_chk = next(c for c in supported_checks if c.target_claim in s or s in c.target_claim)
                        chunk = next((ck for ck in verified_chunks if ck.chunk_id == matching_chk.supporting_chunk_id), verified_chunks[0])
                        title = chunk.metadata.title or chunk.doc_id
                        s = f"{s} [Doc: {title} | Chunk: {chunk.chunk_index} | Hash: {chunk.content_hash[:8]}]"
                    valid_sentences.append(s)

            revised_response = " ".join(valid_sentences) if valid_sentences else baseline_response

        grounding_score = len(supported_checks) / max(1, len(checks)) if checks else 1.0

        return CoVResult(
            baseline_response=baseline_response,
            verification_checks=checks,
            revised_response=revised_response,
            unsupported_claims_removed=unsupported_count,
            grounding_score=round(grounding_score, 4),
            telemetry={
                "total_claims_checked": len(checks),
                "supported_claims": len(supported_checks),
                "unsupported_claims": unsupported_count
            }
        )
