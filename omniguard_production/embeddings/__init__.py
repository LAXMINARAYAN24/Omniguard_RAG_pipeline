"""
omniguard_production.embeddings — Multi-model embedding abstraction & density normalizers.
"""
from .base import EmbeddingProvider
from .neural_provider import DenseNeuralEmbeddingProvider
from .density_normalizer import DensityNormalizer

__all__ = [
    "EmbeddingProvider",
    "DenseNeuralEmbeddingProvider",
    "DensityNormalizer",
]
