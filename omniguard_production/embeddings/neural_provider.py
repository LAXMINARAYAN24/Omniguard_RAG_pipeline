"""
neural_provider.py — Concrete Embedding Providers for Production & Offline Zero-Dependency Deployments.
"""
from __future__ import annotations
import hashlib
import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import HashingVectorizer
from .base import EmbeddingProvider


class DenseNeuralEmbeddingProvider(EmbeddingProvider):
    """Dense embedding provider that supports real neural sentence transformers

    with an integrated deterministic, dense subword projection fallback.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384, fallback_vocab: Optional[List[str]] = None):
        self._model_name = model_name
        self._dim = dim
        self._version = "2.0.0"
        self._neural_model = None
        self._hasher: Optional[HashingVectorizer] = None
        self._projection_matrix: Optional[np.ndarray] = None

        # Attempt to load sentence-transformers if available
        try:
            from sentence_transformers import SentenceTransformer
            from ..config import HF_TOKEN
            if HF_TOKEN:
                self._neural_model = SentenceTransformer(model_name, token=HF_TOKEN)
            else:
                self._neural_model = SentenceTransformer(model_name)
            if hasattr(self._neural_model, "get_embedding_dimension"):
                self._dim = self._neural_model.get_embedding_dimension()
            else:
                self._dim = self._neural_model.get_sentence_embedding_dimension()
        except Exception:
            self._neural_model = None
            self._init_dense_fallback(dim)

    def _init_dense_fallback(self, dim: int):
        """Initializes a deterministic subword hashing projector with gaussian random projection."""
        self._dim = dim
        hash_features = max(1024, dim * 4)
        self._hasher = HashingVectorizer(
            n_features=hash_features,
            alternate_sign=True,
            ngram_range=(1, 3),
            analyzer="char_wb"
        )
        # Deterministic orthonormal / Gaussian random projection matrix
        rng = np.random.RandomState(42)
        proj = rng.randn(hash_features, dim)
        q, _ = np.linalg.qr(proj)
        self._projection_matrix = q.astype(np.float64)

    def fit_corpus(self, texts: List[str]):
        """No-op for hashing projector; maintains deterministic projection."""
        pass

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if self._neural_model is not None:
            try:
                embeddings = self._neural_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings.astype(np.float64)
            except Exception:
                pass

        if self._hasher is None or self._projection_matrix is None:
            self._init_dense_fallback(self._dim)

        sparse_X = self._hasher.transform(texts)
        dense_proj = sparse_X.dot(self._projection_matrix)

        # Unit L2 normalize
        norms = np.linalg.norm(dense_proj, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (dense_proj / norms).astype(np.float64)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return self._version
