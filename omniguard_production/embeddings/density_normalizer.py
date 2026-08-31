"""
density_normalizer.py — Variance & Density Normalizer for Heterogeneous Embeddings.

Standardizes feature variances and scales density across dense/sparse representations
to ensure consistent spectral SVD filter behavior.
"""
from __future__ import annotations
import numpy as np
from typing import Optional


class DensityNormalizer:
    """Normalizes arbitrary embedding vectors into zero-mean, variance-scaled representations."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "DensityNormalizer":
        """Calculates dimension-wise mean and variance on clean calibration data."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std < self.eps] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms vectors into standardized variance space."""
        if self.mean is None or self.std is None:
            self.fit(X)
        normalized = (X - self.mean) / self.std
        # Re-apply L2 normalization per vector
        norms = np.linalg.norm(normalized, axis=-1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return normalized / norms

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
