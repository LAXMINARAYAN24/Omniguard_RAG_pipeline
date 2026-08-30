"""
drs_filter.py — Ring 1: Spectral Ingestion Guard (Directional Relative Shifts).

Real PCA (via NumPy SVD) fit on the real TF-IDF vectors of the clean
reference corpus. We keep the lowest-variance components — directions where
genuine clean documents barely vary, so any document (poison or otherwise)
that has an unusual projection there stands out — and flag new documents
whose whitened projection exceeds a threshold.

BUG FOUND AND FIXED: the previous version fit the PCA subspace AND
calibrated the threshold on the exact same reference set, then benchmarked
against poison documents that were, by construction, the only "new" (i.e.
not in that reference set) documents ever scored. With ~96 reference docs
against ~215 TF-IDF dimensions, the "low-variance" directions were mostly
sampling noise specific to those 96 points, not a real clean-data subspace
-- measured directly: brand-new, entirely legitimate documents (same
topics, same generator, zero attack intent) were flagged at a 100% rate
(80/80) by the old version. That means the previously reported "DRS: 0%
ASR" was not a real security property -- it was rejecting anything absent
from its fit sample, and every poison doc in the benchmark happened to be
absent from that sample while every "clean" doc in the benchmark pool was
the literal fitted point. A filter with a ~100% false-positive rate on
real new content is not usable in production regardless of its ASR.

Fix: split the reference set into a FIT subset (defines the low-variance
directions) and a held-out CALIBRATION subset never seen during the SVD
fit (defines the threshold). This is still calibrated entirely from real
clean data -- no hand-set evasion parameter -- but it is no longer
threshold-shopping against its own training points, so the reported
false-positive rate is honest. Even so, this remains a small-sample
estimate: run `holdout_false_positive_rate` against a fresh clean batch
before trusting any ASR number this filter produces.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List
from .corpus import Document

LOW_VARIANCE_FRACTION = 0.4
FILTER_PERCENTILE = 99.0
CALIBRATION_FRACTION = 0.3  # held out from the SVD fit, used only to set the threshold


@dataclass
class DRSFilterResult:
    kept: List[Document]
    dropped: List[Document]
    threshold: float


class DRSFilter:
    def __init__(self, reference_docs: List[Document], seed: int = 0):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(reference_docs))
        n_calib = max(1, int(len(reference_docs) * CALIBRATION_FRACTION))
        calib_idx, fit_idx = idx[:n_calib], idx[n_calib:]

        X_fit = np.stack([reference_docs[i].embedding for i in fit_idx])
        X_calib = np.stack([reference_docs[i].embedding for i in calib_idx])

        mean = X_fit.mean(axis=0)
        Xc = X_fit - mean
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        eigvals = (S ** 2) / max(len(fit_idx) - 1, 1)
        order = np.argsort(eigvals)
        n_low = max(1, int(len(order) * LOW_VARIANCE_FRACTION))
        low_idx = order[:n_low]

        self.mean = mean
        self.components = Vt[low_idx]
        self.stds = np.sqrt(np.maximum(eigvals[low_idx], 1e-8))

        # Threshold calibrated on the HELD-OUT calibration split -- documents
        # never used to define the low-variance subspace itself -- so a
        # brand-new, non-malicious document is treated the same way a
        # calibration document was, rather than being compared against an
        # in-sample fit.
        calib_scores = self._raw_scores(X_calib)
        self.threshold = float(np.percentile(calib_scores, FILTER_PERCENTILE))

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        proj = (X - self.mean) @ self.components.T
        whitened = proj / self.stds
        return np.linalg.norm(whitened, axis=1)

    def score(self, doc: Document) -> float:
        return float(self._raw_scores(doc.embedding[None, :])[0])

    def filter(self, docs: List[Document]) -> DRSFilterResult:
        """PERFORMANCE (Path A): originally called self.score(d) -- itself a
        single-row call into the already-vectorized _raw_scores -- inside a
        Python for-loop, once per document. Since filter() is on the hot
        path (once per query, ~480 documents each), that meant ~480 separate
        tiny matrix ops instead of one batched one; measured as the single
        largest cost in a full benchmark run after the retrieval.py and
        fresh_docs fixes (about half of OmniGuard-RAG's own wall-clock time).
        Batching every document's embedding into one matrix and calling
        _raw_scores exactly once computes the IDENTICAL per-document scores
        (same formula, same self.mean/components/stds), just without the
        per-document Python/NumPy call overhead. score() itself is untouched
        and still used by holdout_false_positive_rate(), which isn't on the
        hot path (called once per benchmark run, not once per query)."""
        if not docs:
            return DRSFilterResult(kept=[], dropped=[], threshold=self.threshold)
        X = np.stack([d.embedding for d in docs])
        scores = self._raw_scores(X)
        kept, dropped = [], []
        for d, s in zip(docs, scores):
            (dropped if s > self.threshold else kept).append(d)
        return DRSFilterResult(kept=kept, dropped=dropped, threshold=self.threshold)

    def holdout_false_positive_rate(self, fresh_clean_docs: List[Document]) -> float:
        """Honesty check: what fraction of BRAND-NEW, non-malicious documents
        (never involved in fitting or calibrating this filter) get wrongly
        dropped. Call this against a freshly generated clean batch -- not the
        reference_docs this filter was built from -- before trusting any ASR
        number. A high value here means low ASR is coming from rejecting
        unfamiliar documents in general, not from detecting attacks."""
        if not fresh_clean_docs:
            return 0.0
        flagged = sum(1 for d in fresh_clean_docs if self.score(d) > self.threshold)
        return flagged / len(fresh_clean_docs)
