"""
hybrid_fusion.py — Reciprocal Rank Fusion (RRF) & Convex Score Blending for Hybrid Retrieval.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Literal
from ..trust.provenance import ProductionChunk


class HybridFusion:
    """Combines dense vector and BM25 sparse lexical retrieval results."""

    def __init__(self, rrf_k: int = 60, dense_weight: float = 0.6, sparse_weight: float = 0.4):
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse_rrf(self, dense_results: List[Tuple[ProductionChunk, float]],
                 sparse_results: List[Tuple[ProductionChunk, float]],
                 top_k: int = 20) -> List[Tuple[ProductionChunk, float]]:
        """Applies Reciprocal Rank Fusion across dense and sparse candidate rankings."""
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, ProductionChunk] = {}

        # Process dense ranks
        for rank, (chunk, score) in enumerate(dense_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            recip = self.dense_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + recip

        # Process sparse ranks
        for rank, (chunk, score) in enumerate(sparse_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            recip = self.sparse_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + recip

        # Sort descending by fused score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_map[cid], score) for cid, score in sorted_chunks[:top_k]]

    def fuse_convex(self, dense_results: List[Tuple[ProductionChunk, float]],
                    sparse_results: List[Tuple[ProductionChunk, float]],
                    top_k: int = 20) -> List[Tuple[ProductionChunk, float]]:
        """Min-max standardizes raw scores and performs linear convex combination."""
        chunk_map: Dict[str, ProductionChunk] = {}
        dense_norm: Dict[str, float] = {}
        sparse_norm: Dict[str, float] = {}

        if dense_results:
            d_scores = [s for _, s in dense_results]
            min_d, max_d = min(d_scores), max(d_scores)
            d_range = max_d - min_d if max_d > min_d else 1.0
            for c, s in dense_results:
                chunk_map[c.chunk_id] = c
                dense_norm[c.chunk_id] = (s - min_d) / d_range

        if sparse_results:
            s_scores = [s for _, s in sparse_results]
            min_s, max_s = min(s_scores), max(s_scores)
            s_range = max_s - min_s if max_s > min_s else 1.0
            for c, s in sparse_results:
                chunk_map[c.chunk_id] = c
                sparse_norm[c.chunk_id] = (s - min_s) / s_range

        all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
        blended: Dict[str, float] = {}

        for cid in all_ids:
            d_val = dense_norm.get(cid, 0.0)
            s_val = sparse_norm.get(cid, 0.0)
            blended[cid] = (self.dense_weight * d_val) + (self.sparse_weight * s_val)

        sorted_blended = sorted(blended.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_map[cid], score) for cid, score in sorted_blended[:top_k]]
