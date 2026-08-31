"""
base.py — Abstract Base Embedding Provider Interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingProvider(ABC):
    """Abstract interface for all vector embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string into a 1D vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of text strings into a 2D matrix (N, D)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of output vectors."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying model."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Version tag for reproducibility."""
        pass
