"""
citation_tracker.py — Deterministic Citation Extractor & Grounding Verification Engine.

Validates that all inline citations in LLM responses map to authentic, retrieved, and verified chunks,
and performs proposition-level NLI entailment audits to ensure cited chunks actually support the text.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Tuple, Optional
from ..trust.provenance import ProductionChunk
from ..claims.nli_verifier import NLIVerifier

_CITATION_REGEX = re.compile(
    r"\[Doc:\s*([^\|]+?)\s*\|\s*Chunk:\s*(\d+)\s*\|\s*Hash:\s*([a-f0-9]+)\]",
    re.IGNORECASE
)


@dataclass
class SentenceCitationAudit:
    sentence: str
    cleaned_sentence: str
    citations: List[str]
    is_syntactically_valid: bool
    is_entailed: bool
    entailment_score: float
    contradiction_score: float
    supporting_chunk_ids: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


@dataclass
class CitationAuditReport:
    total_citations: int
    valid_citations: int
    invalid_citations: int
    citation_precision: float
    citation_recall: float
    grounding_ratio: float
    citation_entailment_precision: float = 1.0
    valid_labels: List[str] = field(default_factory=list)
    invalid_labels: List[str] = field(default_factory=list)
    sentence_audits: List[SentenceCitationAudit] = field(default_factory=list)
    is_fully_grounded: bool = False
    audit_telemetry: Dict[str, Any] = field(default_factory=dict)


class CitationTracker:
    """Verifies that generated responses strictly cite authentic chunks and entail their claims."""

    def __init__(self, nli_verifier: Optional[NLIVerifier] = None, min_entailment_threshold: float = 0.50):
        self.nli_verifier = nli_verifier or NLIVerifier()
        self.min_entailment_threshold = min_entailment_threshold

    def audit_response(self,
                       generated_text: str,
                       allowed_chunks: List[ProductionChunk],
                       verify_semantic_entailment: bool = True) -> CitationAuditReport:
        """
        Audits all citation markers in the response against authentic chunks and
        verifies proposition-level entailment for each cited sentence.
        """
        if not generated_text.strip():
            return CitationAuditReport(
                total_citations=0,
                valid_citations=0,
                invalid_citations=0,
                citation_precision=1.0,
                citation_recall=1.0 if not allowed_chunks else 0.0,
                grounding_ratio=1.0,
                citation_entailment_precision=1.0,
                is_fully_grounded=True
            )

        matches = _CITATION_REGEX.findall(generated_text)
        total = len(matches)

        # Build chunk lookup maps
        chunk_by_hash: Dict[str, ProductionChunk] = {}
        valid_chunk_hashes: Set[str] = set()
        valid_chunk_labels: Set[str] = set()

        for c in allowed_chunks:
            title = c.metadata.title or c.doc_id
            h_short = c.content_hash[:8].lower()
            valid_chunk_hashes.add(h_short)
            chunk_by_hash[h_short] = c
            valid_chunk_labels.add(f"[Doc: {title} | Chunk: {c.chunk_index} | Hash: {h_short}]")

        valid_found: List[str] = []
        invalid_found: List[str] = []

        for doc_title, chunk_idx_str, hash_prefix in matches:
            label = f"[Doc: {doc_title.strip()} | Chunk: {chunk_idx_str.strip()} | Hash: {hash_prefix.strip().lower()}]"
            h_clean = hash_prefix.strip().lower()

            if h_clean in valid_chunk_hashes:
                valid_found.append(label)
            else:
                invalid_found.append(label)

        num_valid = len(valid_found)
        num_invalid = len(invalid_found)
        precision = num_valid / max(1, total) if total > 0 else 1.0

        # Unique valid chunks referenced
        unique_valid_hashes = set(l.split("Hash: ")[1].rstrip("]").strip() for l in valid_found if "Hash: " in l)
        recall = len(unique_valid_hashes) / max(1, len(allowed_chunks)) if allowed_chunks else 1.0

        # Sentence-level grounding and semantic entailment audit
        # Split on sentence boundaries, keeping citations attached to their respective sentence
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?\]])\s+(?!\[Doc:)", generated_text) if len(s.strip()) > 5]
        sentence_audits: List[SentenceCitationAudit] = []
        grounded_sents = 0
        entailed_citations_count = 0
        total_valid_citation_checks = 0

        for sent in raw_sentences:
            s_matches = _CITATION_REGEX.findall(sent)
            cleaned_s = _CITATION_REGEX.sub("", sent).strip()
            if not cleaned_s:
                continue

            s_valid = True
            s_entailed = False
            best_ent = 0.0
            best_contra = 0.0
            supp_chunks = []
            flags = []

            if not s_matches:
                flags.append("MISSING_CITATION")
                sentence_audits.append(SentenceCitationAudit(
                    sentence=sent,
                    cleaned_sentence=cleaned_s,
                    citations=[],
                    is_syntactically_valid=True,
                    is_entailed=False,
                    entailment_score=0.0,
                    contradiction_score=0.0,
                    supporting_chunk_ids=[],
                    flags=flags
                ))
                continue

            # Evaluate each citation in sentence
            for doc_t, c_idx, h_prefix in s_matches:
                h_c = h_prefix.strip().lower()
                if h_c not in valid_chunk_hashes:
                    s_valid = False
                    flags.append(f"INVALID_HASH_{h_c}")
                else:
                    target_chunk = chunk_by_hash[h_c]
                    supp_chunks.append(target_chunk.chunk_id)
                    total_valid_citation_checks += 1

                    if verify_semantic_entailment and self.nli_verifier is not None:
                        # Verify premise (chunk text) entails hypothesis (cleaned sentence)
                        nli_res = self.nli_verifier.check_pair(target_chunk.clean_text, cleaned_s)
                        ent = nli_res.get("entailment", 0.0)
                        contra = nli_res.get("contradiction", 0.0)
                        if ent > best_ent:
                            best_ent = ent
                            best_contra = contra

                        if ent >= self.min_entailment_threshold and contra < 0.30:
                            s_entailed = True
                            entailed_citations_count += 1
                        else:
                            flags.append(f"WEAK_ENTAILMENT_{h_c}_({ent:.2f})")
                    else:
                        s_entailed = True
                        entailed_citations_count += 1
                        best_ent = 1.0

            if s_valid and (s_entailed or not verify_semantic_entailment):
                grounded_sents += 1

            sentence_audits.append(SentenceCitationAudit(
                sentence=sent,
                cleaned_sentence=cleaned_s,
                citations=[f"[Doc: {dt} | Chunk: {ci} | Hash: {hp}]" for dt, ci, hp in s_matches],
                is_syntactically_valid=s_valid,
                is_entailed=s_entailed,
                entailment_score=round(best_ent, 4),
                contradiction_score=round(best_contra, 4),
                supporting_chunk_ids=supp_chunks,
                flags=flags
            ))

        grounding_ratio = grounded_sents / max(1, len(sentence_audits)) if sentence_audits else 1.0
        entailment_prec = entailed_citations_count / max(1, total_valid_citation_checks) if total_valid_citation_checks > 0 else 1.0
        is_grounded = (num_invalid == 0 and (total > 0 or len(allowed_chunks) == 0) and entailment_prec >= 0.60)

        return CitationAuditReport(
            total_citations=total,
            valid_citations=num_valid,
            invalid_citations=num_invalid,
            citation_precision=round(precision, 4),
            citation_recall=round(recall, 4),
            grounding_ratio=round(grounding_ratio, 4),
            citation_entailment_precision=round(entailment_prec, 4),
            valid_labels=valid_found,
            invalid_labels=invalid_found,
            sentence_audits=sentence_audits,
            is_fully_grounded=is_grounded,
            audit_telemetry={
                "total_sentences": len(sentence_audits),
                "grounded_sentences": grounded_sents,
                "entailed_citations_count": entailed_citations_count,
                "total_valid_citation_checks": total_valid_citation_checks
            }
        )
