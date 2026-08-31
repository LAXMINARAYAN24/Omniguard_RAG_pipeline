"""
claim_extractor.py — Atomic Proposition Extraction from Ingested Chunks & Generated Text.

Decomposes complex, compound passages into discrete, verifiable atomic claims.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from ..trust.provenance import ProductionChunk


@dataclass
class AtomicClaim:
    """Represents a single atomic factual proposition."""
    claim_id: str
    text: str
    source_chunk_id: Optional[str] = None
    subject: str = ""
    predicate: str = ""
    target_object: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Sentence splitting regex respecting common abbreviations
_SENTENCE_SPLIT_REGEX = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+")
# Conjunction clause splitting patterns
_CLAUSE_SPLIT_REGEX = re.compile(r"\b(?:whereas|while|although|however|moreover|furthermore|additionally|but)\b", re.IGNORECASE)


class ClaimExtractor:
    """Extracts atomic factual claims using rule-based semantic parsing with LLM augmentation hook."""

    def __init__(self, max_claims_per_chunk: int = 10):
        self.max_claims_per_chunk = max_claims_per_chunk

    def extract_from_chunk(self, chunk: ProductionChunk) -> List[AtomicClaim]:
        """Extracts atomic claims from a single production chunk."""
        return self.extract_from_text(chunk.clean_text, source_chunk_id=chunk.chunk_id)

    def extract_from_text(self, text: str, source_chunk_id: Optional[str] = None) -> List[AtomicClaim]:
        """Decomposes raw text into individual atomic propositions."""
        if not text or not text.strip():
            return []

        # Strip citation annotations and metadata tags before sentence splitting
        cleaned_text = re.sub(r"\[Doc:[^\]]*\]", "", text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\[(?:Chunk|Hash|Source|Ref):[^\]]*\]", "", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\[\d+\]", "", cleaned_text)
        cleaned_text = re.sub(r"\bref\s*#?\d+\b", "", cleaned_text, flags=re.IGNORECASE)

        raw_sentences = _SENTENCE_SPLIT_REGEX.split(cleaned_text.strip())
        claims: List[AtomicClaim] = []
        counter = 0

        for s_idx, sent in enumerate(raw_sentences):
            sent = sent.strip()
            if not sent or len(sent) < 8:
                continue

            # Split compound clauses on strong contrasting conjunctions
            clauses = _CLAUSE_SPLIT_REGEX.split(sent)
            for clause in clauses:
                clause = clause.strip().rstrip(",;:- ")
                # Strip reference markers like 'ref #1', '[1]', '(ref 2)', '[Doc: ...]'
                clean_clause = re.sub(r"\[.*?\]", "", clause)
                clean_clause = re.sub(r"\bref\s*#?\d+\b", "", clean_clause, flags=re.IGNORECASE)
                clean_clause = re.sub(r"\s+", " ", clean_clause).strip().rstrip(",;:- ")
                if not clean_clause or len(clean_clause.split()) < 3:
                    continue

                counter += 1
                claim_id = f"{source_chunk_id or 'doc'}_c{counter}"

                # Rule-based subject/predicate identification
                subject, pred, obj = self._parse_spo(clean_clause)

                claims.append(AtomicClaim(
                    claim_id=claim_id,
                    text=clean_clause,
                    source_chunk_id=source_chunk_id,
                    subject=subject,
                    predicate=pred,
                    target_object=obj,
                    confidence=0.95
                ))

                if len(claims) >= self.max_claims_per_chunk:
                    return claims

        return claims

    def _parse_spo(self, clause: str) -> tuple[str, str, str]:
        """Basic lightweight subject-verb-object extractor for proposition tagging."""
        tokens = clause.split()
        if len(tokens) <= 3:
            return tokens[0] if tokens else "", "is", " ".join(tokens[1:])

        # Find potential main verbs / linking words
        verb_indices = [
            i for i, t in enumerate(tokens)
            if t.lower() in {"is", "are", "was", "were", "causes", "produces", "uses", "contains",
                             "transmits", "requires", "operates", "connects", "synthesizes", "results"}
        ]

        if verb_indices:
            v_idx = verb_indices[0]
            subj = " ".join(tokens[:v_idx])
            pred = tokens[v_idx]
            obj = " ".join(tokens[v_idx+1:])
            return subj, pred, obj

        return tokens[0], "relates_to", " ".join(tokens[1:])
