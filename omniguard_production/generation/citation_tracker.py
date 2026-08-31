"""
citation_tracker.py — Deterministic Citation Extractor & Grounding Verification Engine.

Validates that all inline citations in LLM responses map to authentic, retrieved, and verified chunks.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Tuple
from ..trust.provenance import ProductionChunk

_CITATION_REGEX = re.compile(
    r"\[Doc:\s*([^\|]+?)\s*\|\s*Chunk:\s*(\d+)\s*\|\s*Hash:\s*([a-f0-9]+)\]",
    re.IGNORECASE
)


@dataclass
class CitationAuditReport:
    total_citations: int
    valid_citations: int
    invalid_citations: int
    citation_precision: float
    citation_recall: float
    grounding_ratio: float
    valid_labels: List[str] = field(default_factory=list)
    invalid_labels: List[str] = field(default_factory=list)
    is_fully_grounded: bool = False


class CitationTracker:
    """Verifies that generated responses strictly cite only authentic retrieved chunks."""

    def __init__(self):
        pass

    def audit_response(self,
                       generated_text: str,
                       allowed_chunks: List[ProductionChunk]) -> CitationAuditReport:
        """Audits all citation markers in the response against authentic chunks."""
        matches = _CITATION_REGEX.findall(generated_text)
        total = len(matches)

        # Build lookup set of valid (title/doc_id, chunk_index, hash_prefix)
        valid_chunk_hashes: Set[str] = set()
        valid_chunk_labels: Set[str] = set()

        for c in allowed_chunks:
            title = c.metadata.title or c.doc_id
            h_short = c.content_hash[:8].lower()
            valid_chunk_hashes.add(h_short)
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
        unique_valid_hashes = set(l.split("Hash: ")[1].rstrip("]").strip() for l in valid_found)
        recall = len(unique_valid_hashes) / max(1, len(allowed_chunks)) if allowed_chunks else 1.0

        # Sentence-level grounding ratio
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", generated_text) if len(s.strip()) > 10]
        grounded_sents = 0
        for s in sentences:
            if _CITATION_REGEX.search(s):
                grounded_sents += 1

        grounding_ratio = grounded_sents / max(1, len(sentences)) if sentences else 1.0
        is_grounded = (num_invalid == 0 and (total > 0 or len(allowed_chunks) == 0))

        return CitationAuditReport(
            total_citations=total,
            valid_citations=num_valid,
            invalid_citations=num_invalid,
            citation_precision=round(precision, 4),
            citation_recall=round(recall, 4),
            grounding_ratio=round(grounding_ratio, 4),
            valid_labels=valid_found,
            invalid_labels=invalid_found,
            is_fully_grounded=is_grounded
        )
