"""
cross_reranker.py — Cross-Encoder Neural Alignment Reranker with Dynamic Adaptive-k Selection.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from ..trust.provenance import ProductionChunk
from ..embeddings.base import EmbeddingProvider


class CrossEncoderReranker:
    """Neural cross-encoder reranker with dynamic adaptive-k retrieval cutoff."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2",
                 embedding_provider: Optional[EmbeddingProvider] = None,
                 min_k: int = 3, max_k: int = 15,
                 confidence_threshold: float = 0.45):
        self.model_name = model_name
        self.embedding_provider = embedding_provider
        self.min_k = min_k
        self.max_k = max_k
        self.confidence_threshold = confidence_threshold
        self._cross_model = None

        try:
            from sentence_transformers import CrossEncoder
            from ..config import HF_TOKEN
            if HF_TOKEN:
                self._cross_model = CrossEncoder(model_name, token=HF_TOKEN)
            else:
                self._cross_model = CrossEncoder(model_name)
        except Exception:
            self._cross_model = None

    def rerank(self, query_text: str,
               candidates: List[ProductionChunk],
               apply_security_penalty: bool = True) -> List[Tuple[ProductionChunk, float, Dict[str, Any]]]:
        """Reranks candidate chunks using full cross-attention interaction."""
        if not candidates:
            return []

        if self._cross_model is not None:
            pairs = [[query_text, c.clean_text] for c in candidates]
            raw_scores = self._cross_model.predict(pairs)
            if isinstance(raw_scores, np.ndarray):
                raw_scores = raw_scores.tolist()
            elif isinstance(raw_scores, (int, float)):
                raw_scores = [float(raw_scores)]
            # Calibrate logits to [0, 1] probability range via numerically stable sigmoid
            raw_scores = [float(1.0 / (1.0 + np.exp(-np.clip(float(s), -20.0, 20.0)))) for s in raw_scores]
        else:
            # Fallback pairwise interaction: semantic similarity + exact lexical coverage
            raw_scores = self._fallback_score_pairs(query_text, candidates)

        results = []
        for chunk, score in zip(candidates, raw_scores):
            sec_penalty = 0.0
            if apply_security_penalty and chunk.security_flags:
                # Deduct security penalty for suspicious patterns without zeroing out raw score
                sec_penalty = 0.25 * len(chunk.security_flags)

            effective_score = max(0.0, float(score) - sec_penalty)
            results.append((
                chunk,
                effective_score,
                {
                    "raw_score": round(float(score), 4),
                    "security_penalty": round(sec_penalty, 4),
                    "effective_score": round(effective_score, 4),
                    "trust_multiplier": chunk.trust_score
                }
            ))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def select_adaptive_k(self, scored_results: List[Tuple[ProductionChunk, float, Dict[str, Any]]]) -> List[Tuple[ProductionChunk, float, Dict[str, Any]]]:
        """Dynamically determines optimal cutoff k based on score dropoff knee and confidence."""
        if not scored_results:
            return []

        n = len(scored_results)
        if n <= self.min_k:
            return scored_results

        scores = [s for _, s, _ in scored_results]
        top_score = scores[0]

        # Retain items that meet confidence threshold and stay within dynamic range of top score
        qualifying_indices = []
        for i, s in enumerate(scores):
            if s >= self.confidence_threshold and (top_score <= 0 or (s / max(1e-6, top_score)) >= 0.65):
                qualifying_indices.append(i)
            elif i < self.min_k:
                qualifying_indices.append(i)

        if not qualifying_indices:
            chosen_k = min(self.min_k, n)
        else:
            chosen_k = min(max(len(qualifying_indices), self.min_k), min(self.max_k, n))

        return scored_results[:chosen_k]

    def _fallback_score_pairs(self, query: str, chunks: List[ProductionChunk]) -> List[float]:
        q_tokens = set(query.lower().split())
        scores = []
        for c in chunks:
            c_tokens = set(c.clean_text.lower().split())
            overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))

            # Embedding cosine if available
            cos_sim = 0.5
            if self.embedding_provider is not None:
                q_emb = self.embedding_provider.embed_text(query)
                if c.embedding is not None:
                    c_emb = c.embedding
                else:
                    c_emb = self.embedding_provider.embed_text(c.clean_text)
                cos_sim = float(np.dot(q_emb, c_emb) / (max(1e-8, np.linalg.norm(q_emb) * np.linalg.norm(c_emb))))
                cos_sim = (cos_sim + 1.0) / 2.0  # scale to [0, 1]

            score = 0.6 * cos_sim + 0.4 * overlap
            scores.append(float(score))
        return scores
