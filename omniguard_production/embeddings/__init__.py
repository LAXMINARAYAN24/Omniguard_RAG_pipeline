"""
omniguard_production.embeddings — Multi-model embedding abstraction & density normalizers.
"""
from .base import EmbeddingProvider
from .neural_provider import DenseNeuralEmbeddingProvider, EmbeddingMode
from .density_normalizer import DensityNormalizer
from .drs_engine import DRSEngine, DRSCalibrator, DRSModel, DRSScoreResult, DRSConfig

__all__ = [
    "EmbeddingProvider",
    "DenseNeuralEmbeddingProvider",
    "EmbeddingMode",
    "DensityNormalizer",
    "DRSEngine",
    "DRSCalibrator",
    "DRSModel",
    "DRSScoreResult",
    "DRSConfig",
]
