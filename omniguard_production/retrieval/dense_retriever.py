"""
dense_retriever.py — High-Performance Vector Similarity Index for Production Chunks.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from ..trust.provenance import ProductionChunk, DocumentState
from ..embeddings.base import EmbeddingProvider


class DenseRetriever:
    """Vector index supporting cosine similarity search with tenant filtering."""

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.chunks: List[ProductionChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._id_to_idx: Dict[str, int] = {}

    def index_chunks(self, chunks: List[ProductionChunk]):
        """Indexes or updates a list of production chunks."""
        texts_to_embed = []
        chunks_to_embed_idx = []

        for c in chunks:
            if c.chunk_id in self._id_to_idx:
                idx = self._id_to_idx[c.chunk_id]
                self.chunks[idx] = c
                if c.embedding is None:
                    texts_to_embed.append(c.clean_text)
                    chunks_to_embed_idx.append(idx)
            else:
                idx = len(self.chunks)
                self._id_to_idx[c.chunk_id] = idx
                self.chunks.append(c)
                if c.embedding is None:
                    texts_to_embed.append(c.clean_text)
                    chunks_to_embed_idx.append(idx)

        if texts_to_embed:
            new_embs = self.embedding_provider.embed_batch(texts_to_embed)
            for idx, emb in zip(chunks_to_embed_idx, new_embs):
                self.chunks[idx].embedding = emb

        # Stack embeddings into continuous matrix
        all_vecs = [c.embedding for c in self.chunks if c.embedding is not None]
        if all_vecs:
            self.embeddings = np.array(all_vecs, dtype=np.float64)
            # Ensure L2 unit norm
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            self.embeddings = self.embeddings / norms
        else:
            self.embeddings = None

    def search(self, query_text: str, top_k: int = 20,
               tenant_id: Optional[str] = None,
               query_embedding: Optional[np.ndarray] = None) -> List[Tuple[ProductionChunk, float]]:
        """Performs cosine similarity search returning top-k matching chunks."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        if query_embedding is None:
            query_embedding = self.embedding_provider.embed_text(query_text)

        q_norm = np.linalg.norm(query_embedding)
        if q_norm > 0:
            query_embedding = query_embedding / q_norm

        sims = np.dot(self.embeddings, query_embedding)

        # Apply state and tenant filters
        valid_indices = []
        for i, c in enumerate(self.chunks):
            if c.state in {DocumentState.ARCHIVED, DocumentState.SUSPICIOUS}:
                continue
            if tenant_id and c.metadata.tenant_id != tenant_id:
                continue
            valid_indices.append(i)

        if not valid_indices:
            return []

        valid_indices = np.array(valid_indices)
        valid_sims = sims[valid_indices]

        # Top-k selection
        k = min(top_k, len(valid_indices))
        top_k_local_idx = np.argpartition(-valid_sims, k - 1)[:k]
        top_k_sorted = top_k_local_idx[np.argsort(-valid_sims[top_k_local_idx])]

        results = []
        for local_idx in top_k_sorted:
            orig_idx = valid_indices[local_idx]
            results.append((self.chunks[orig_idx], float(valid_sims[local_idx])))

        return results
