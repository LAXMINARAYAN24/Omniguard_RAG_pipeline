"""
neural_provider.py — Concrete Embedding Providers for Production & Offline Deployments.

Features:
1. Strict enterprise fail-closed mode (`EmbeddingMode.STRICT`): fails immediately if neural models cannot be initialized or fail to encode, preventing silent security geometry degradation.
2. Explicit degraded fallback mode (`EmbeddingMode.DEGRADED_FALLBACK`): deterministic subword random-projection fallback with observable telemetry flags.
3. Full observability: exposes model status, device, dimension, and fallback indicators.
"""
from __future__ import annotations
import enum
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.feature_extraction.text import HashingVectorizer
from .base import EmbeddingProvider

logger = logging.getLogger("omniguard.neural_provider")


class EmbeddingMode(str, enum.Enum):
    STRICT = "STRICT"                      # Fail-closed if real neural model is unavailable
    DEGRADED_FALLBACK = "DEGRADED_FALLBACK"  # Explicitly allow deterministic hash/random projection


class DenseNeuralEmbeddingProvider(EmbeddingProvider):
    """Dense embedding provider supporting SentenceTransformers with strict fail-closed governance."""

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 dim: int = 384,
                 mode: EmbeddingMode = EmbeddingMode.DEGRADED_FALLBACK,
                 fallback_vocab: Optional[List[str]] = None):
        self._model_name = model_name
        self._dim = dim
        self._mode = mode
        self._version = "2.1.0"
        self._neural_model = None
        self._is_fallback = False
        self._hasher: Optional[HashingVectorizer] = None
        self._projection_matrix: Optional[np.ndarray] = None
        self._initialization_error: Optional[str] = None

        # Attempt to load sentence-transformers
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

            logger.info(f"Loaded neural embedding model '{model_name}' (dim={self._dim})")
        except Exception as e:
            self._neural_model = None
            self._initialization_error = str(e)
            if self._mode == EmbeddingMode.STRICT:
                raise RuntimeError(
                    f"EmbeddingProvider in STRICT mode failed to load neural model '{model_name}'. "
                    f"Fail-closed policy activated: {e}"
                )
            logger.warning(
                f"Neural model '{model_name}' unavailable ({e}). "
                f"Operating in explicit DEGRADED_FALLBACK mode (dim={dim})."
            )
            self._is_fallback = True
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
        rng = np.random.RandomState(42)
        proj = rng.randn(hash_features, dim)
        q, _ = np.linalg.qr(proj)
        self._projection_matrix = q.astype(np.float64)
        self._is_fallback = True

    def fit_corpus(self, texts: List[str]):
        """No-op for hashing projector; maintains deterministic projection."""
        pass

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float64)

        if self._neural_model is not None:
            try:
                embeddings = self._neural_model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return embeddings.astype(np.float64)
            except Exception as e:
                if self._mode == EmbeddingMode.STRICT:
                    raise RuntimeError(f"Strict neural encoding failed for batch of {len(texts)} texts: {e}")
                logger.warning(f"Neural encoding failed ({e}); falling back to deterministic projection.")
                self._is_fallback = True

        if self._hasher is None or self._projection_matrix is None:
            if self._mode == EmbeddingMode.STRICT:
                raise RuntimeError("Cannot use fallback projection in STRICT embedding mode.")
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

    @property
    def is_fallback(self) -> bool:
        """True if operating in degraded fallback projection rather than genuine neural transformer."""
        return self._is_fallback

    @property
    def mode(self) -> EmbeddingMode:
        return self._mode

    def get_telemetry_status(self) -> Dict[str, Any]:
        """Returns observable telemetry dictionary regarding embedding provider health."""
        return {
            "model_name": self._model_name,
            "dimension": self._dim,
            "mode": self._mode.value,
            "is_neural": self._neural_model is not None,
            "is_fallback": self._is_fallback,
            "initialization_error": self._initialization_error,
            "version": self._version,
        }
