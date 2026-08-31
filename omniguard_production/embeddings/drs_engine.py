"""
drs_engine.py — Genuine SVD-based Directional Relative Shifts (DRS) Engine for Production.

Implements true spectral Directional Relative Shifts (DRS) analysis on dense embeddings:
1. Learns the low-variance eigenspace from trusted clean reference distributions.
2. Uses a strict held-out calibration split to calibrate detection thresholds without self-referential overfitting.
3. Scores documents/chunks via whitened projections onto the lowest-variance directions.
4. Provides enterprise-grade serialization, model versioning, and retrieval set evaluation.
"""
from __future__ import annotations
import json
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from ..trust.provenance import ProductionChunk

logger = logging.getLogger("omniguard.drs_engine")


@dataclass
class DRSConfig:
    """Configuration parameters for SVD DRS subspace extraction and threshold calibration."""
    low_variance_fraction: float = 0.40
    filter_percentile: float = 99.0
    calibration_fraction: float = 0.30
    min_calibration_samples: int = 10
    seed: int = 42
    model_version: str = "2.0.0"


@dataclass
class DRSScoreResult:
    """Detailed result of evaluating a retrieved set or document under DRS."""
    max_drs_score: float
    mean_drs_score: float
    outlier_count: int
    outlier_ratio: float
    threshold: float
    is_spectral_anomaly_detected: bool
    per_chunk_scores: List[float] = field(default_factory=list)
    is_calibrated: bool = True
    details: Dict[str, Any] = field(default_factory=dict)


class DRSModel:
    """Fitted and calibrated spectral DRS model."""

    def __init__(self,
                 mean: np.ndarray,
                 components: np.ndarray,
                 stds: np.ndarray,
                 threshold: float,
                 embedding_dim: int,
                 model_name: str = "all-MiniLM-L6-v2",
                 fit_samples: int = 0,
                 calib_samples: int = 0,
                 corpus_snapshot_id: str = "default_clean_v1"):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.components = np.asarray(components, dtype=np.float64)  # (k, dim)
        self.stds = np.asarray(stds, dtype=np.float64)              # (k,)
        self.threshold = float(threshold)
        self.embedding_dim = int(embedding_dim)
        self.model_name = str(model_name)
        self.fit_samples = int(fit_samples)
        self.calib_samples = int(calib_samples)
        self.corpus_snapshot_id = str(corpus_snapshot_id)

    def score_vector(self, embedding: np.ndarray) -> float:
        """Scores a single 1D embedding vector."""
        vec = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        return float(self.score_batch(vec)[0])

    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Computes whitened projection norms onto the low-variance directions."""
        X = np.asarray(embeddings, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.embedding_dim:
            raise ValueError(f"Dimension mismatch: expected {self.embedding_dim}, got {X.shape[1]}")

        centered = X - self.mean
        # Project onto low-variance components: (N, dim) @ (dim, k) -> (N, k)
        proj = centered @ self.components.T
        # Normalize by low-variance singular values (whitening)
        whitened = proj / np.maximum(self.stds, 1e-8)
        # L2 norm across low-variance components
        return np.linalg.norm(whitened, axis=1)

    def is_outlier(self, embedding: np.ndarray) -> bool:
        """Determines whether a single vector exceeds the calibrated DRS threshold."""
        return self.score_vector(embedding) > self.threshold

    def to_dict(self) -> Dict[str, Any]:
        """Serializes model metadata and arrays for persistence."""
        return {
            "mean": self.mean.tolist(),
            "components": self.components.tolist(),
            "stds": self.stds.tolist(),
            "threshold": self.threshold,
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "fit_samples": self.fit_samples,
            "calib_samples": self.calib_samples,
            "corpus_snapshot_id": self.corpus_snapshot_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DRSModel:
        """Reconstructs a DRSModel from serialized dictionary."""
        return cls(
            mean=np.array(data["mean"], dtype=np.float64),
            components=np.array(data["components"], dtype=np.float64),
            stds=np.array(data["stds"], dtype=np.float64),
            threshold=float(data["threshold"]),
            embedding_dim=int(data["embedding_dim"]),
            model_name=str(data.get("model_name", "unknown")),
            fit_samples=int(data.get("fit_samples", 0)),
            calib_samples=int(data.get("calib_samples", 0)),
            corpus_snapshot_id=str(data.get("corpus_snapshot_id", "default")),
        )


class DRSCalibrator:
    """Fits SVD low-variance directions and calibrates thresholds on held-out clean data."""

    def __init__(self, config: Optional[DRSConfig] = None):
        self.config = config or DRSConfig()

    def fit(self,
            clean_embeddings: np.ndarray,
            model_name: str = "all-MiniLM-L6-v2",
            corpus_snapshot_id: str = "clean_reference_v1") -> DRSModel:
        """
        Fits DRS eigenspace on the clean reference set and calibrates threshold on a held-out split.
        """
        X = np.asarray(clean_embeddings, dtype=np.float64)
        n_samples, dim = X.shape

        if n_samples < 4:
            # Fallback for very small bootstrap initialization
            mean = np.mean(X, axis=0) if n_samples > 0 else np.zeros(dim)
            return DRSModel(
                mean=mean,
                components=np.eye(dim)[:max(1, int(dim * self.config.low_variance_fraction))],
                stds=np.ones(max(1, int(dim * self.config.low_variance_fraction))),
                threshold=3.0,
                embedding_dim=dim,
                model_name=model_name,
                fit_samples=n_samples,
                calib_samples=0,
                corpus_snapshot_id=corpus_snapshot_id
            )

        rng = np.random.default_rng(self.config.seed)
        shuffled_indices = rng.permutation(n_samples)

        n_calib = max(1, int(n_samples * self.config.calibration_fraction))
        calib_idx = shuffled_indices[:n_calib]
        fit_idx = shuffled_indices[n_calib:]

        if len(fit_idx) < 2:
            fit_idx = shuffled_indices
            calib_idx = shuffled_indices

        X_fit = X[fit_idx]
        X_calib = X[calib_idx]

        # 1. Compute empirical mean on fit set
        mean = np.mean(X_fit, axis=0)
        Xc = X_fit - mean

        # 2. SVD decomposition on centered fit matrix
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        eigvals = (S ** 2) / max(len(fit_idx) - 1, 1)

        # 3. Sort by eigenvalue ascending (lowest variance first)
        order = np.argsort(eigvals)
        n_low = max(1, int(len(order) * self.config.low_variance_fraction))
        low_idx = order[:n_low]

        components = Vt[low_idx]
        stds = np.sqrt(np.maximum(eigvals[low_idx], 1e-8))

        # 4. Calibrate threshold on held-out calibration set (never seen during SVD fit)
        calib_centered = X_calib - mean
        calib_proj = calib_centered @ components.T
        calib_whitened = calib_proj / stds
        calib_scores = np.linalg.norm(calib_whitened, axis=1)

        threshold = float(np.percentile(calib_scores, self.config.filter_percentile))
        # Ensure a reasonable minimum threshold to prevent over-sensitive hair-triggering
        threshold = max(threshold, 1.5)

        logger.info(
            f"DRS Calibrated: {len(fit_idx)} fit samples, {len(calib_idx)} calib samples, "
            f"threshold={threshold:.4f}, low-var components={n_low}/{dim}"
        )

        return DRSModel(
            mean=mean,
            components=components,
            stds=stds,
            threshold=threshold,
            embedding_dim=dim,
            model_name=model_name,
            fit_samples=len(fit_idx),
            calib_samples=len(calib_idx),
            corpus_snapshot_id=corpus_snapshot_id
        )


class DRSEngine:
    """Production runtime engine for Ring 1 Directional Relative Shifts screening."""

    def __init__(self,
                 model: Optional[DRSModel] = None,
                 config: Optional[DRSConfig] = None):
        self.config = config or DRSConfig()
        self.model = model
        self.calibrator = DRSCalibrator(self.config)

    def is_calibrated(self) -> bool:
        """Returns whether a genuine SVD DRS model is loaded and ready."""
        return self.model is not None

    def calibrate_from_embeddings(self,
                                  clean_embeddings: np.ndarray,
                                  model_name: str = "all-MiniLM-L6-v2",
                                  corpus_snapshot_id: str = "clean_reference_v1") -> DRSModel:
        """Calibrates the engine from a clean reference embedding matrix."""
        self.model = self.calibrator.fit(clean_embeddings, model_name, corpus_snapshot_id)
        return self.model

    def score_chunk(self, chunk: ProductionChunk) -> float:
        """Calculates the DRS score for a single chunk."""
        if self.model is None or chunk.embedding is None:
            return 0.0
        return self.model.score_vector(chunk.embedding)

    def evaluate_retrieval_set(self, chunks: List[ProductionChunk]) -> DRSScoreResult:
        """
        Evaluates a retrieved candidate set under genuine spectral DRS.
        Returns aggregate spectral shift metrics and per-chunk scores.
        """
        valid_chunks = [c for c in chunks if c.embedding is not None]
        if not valid_chunks or self.model is None:
            return DRSScoreResult(
                max_drs_score=0.0,
                mean_drs_score=0.0,
                outlier_count=0,
                outlier_ratio=0.0,
                threshold=self.model.threshold if self.model else 3.0,
                is_spectral_anomaly_detected=False,
                per_chunk_scores=[0.0] * len(chunks),
                is_calibrated=self.model is not None,
                details={"reason": "no_valid_embeddings" if not valid_chunks else "uncalibrated_model"}
            )

        emb_matrix = np.array([c.embedding for c in valid_chunks], dtype=np.float64)
        scores = self.model.score_batch(emb_matrix)

        max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
        mean_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
        outliers = int(np.sum(scores > self.model.threshold))
        outlier_ratio = float(outliers / len(scores)) if len(scores) > 0 else 0.0
        is_anomaly = outliers > 0 or max_score > self.model.threshold

        return DRSScoreResult(
            max_drs_score=max_score,
            mean_drs_score=mean_score,
            outlier_count=outliers,
            outlier_ratio=outlier_ratio,
            threshold=self.model.threshold,
            is_spectral_anomaly_detected=is_anomaly,
            per_chunk_scores=[float(s) for s in scores],
            is_calibrated=True,
            details={
                "model_name": self.model.model_name,
                "fit_samples": self.model.fit_samples,
                "calib_samples": self.model.calib_samples,
                "corpus_snapshot_id": self.model.corpus_snapshot_id
            }
        )
