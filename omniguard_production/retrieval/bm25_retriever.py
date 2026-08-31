"""
bm25_retriever.py — High-Efficiency In-Memory BM25 Okapi Lexical Retriever.
"""
from __future__ import annotations
import math
import re
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter, defaultdict
from ..trust.provenance import ProductionChunk, DocumentState

_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


class BM25Retriever:
    """Okapi BM25 index over production document chunks with tenant isolation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[ProductionChunk] = []
        self._doc_lens: List[int] = []
        self._avgdl: float = 0.0
        self._inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # term -> [(chunk_idx, tf)]
        self._doc_frequencies: Dict[str, int] = defaultdict(int)
        self._id_to_idx: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in _TOKEN_PATTERN.findall(text)]

    def index_chunks(self, chunks: List[ProductionChunk]):
        """Indexes a list of chunks, building the inverted index and term statistics."""
        self.chunks = list(chunks)
        self._doc_lens = []
        self._inverted_index.clear()
        self._doc_frequencies.clear()
        self._id_to_idx.clear()

        total_len = 0
        for idx, chunk in enumerate(self.chunks):
            self._id_to_idx[chunk.chunk_id] = idx
            tokens = self._tokenize(chunk.clean_text)
            doc_len = len(tokens)
            self._doc_lens.append(doc_len)
            total_len += doc_len

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self._inverted_index[term].append((idx, count))
                self._doc_frequencies[term] += 1

        self._avgdl = total_len / max(1, len(self.chunks))

    def search(self, query_text: str, top_k: int = 20,
               tenant_id: Optional[str] = None) -> List[Tuple[ProductionChunk, float]]:
        """Scores all eligible chunks against query terms using standard BM25 Okapi."""
        if not self.chunks:
            return []

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return []

        n_docs = len(self.chunks)
        scores: Dict[int, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self._inverted_index:
                continue

            df = self._doc_frequencies[term]
            # Standard Robertson-Spärck Jones IDF
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            if idf <= 0:
                idf = 1e-4

            for chunk_idx, tf in self._inverted_index[term]:
                doc_len = self._doc_lens[chunk_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-4, self._avgdl)))
                score_term = idf * (tf * (self.k1 + 1.0)) / max(1e-4, denom)
                scores[chunk_idx] += score_term

        # Filter chunks by status and tenant
        valid_scores = []
        for chunk_idx, score in scores.items():
            chunk = self.chunks[chunk_idx]
            if chunk.state in {DocumentState.ARCHIVED, DocumentState.SUSPICIOUS}:
                continue
            if tenant_id and chunk.metadata.tenant_id != tenant_id:
                continue
            valid_scores.append((chunk, score))

        valid_scores.sort(key=lambda x: x[1], reverse=True)
        return valid_scores[:top_k]
