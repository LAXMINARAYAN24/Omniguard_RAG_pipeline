"""
query_guard.py — Ring 0: Query-Path Guard.

Real keyword-stuffing suffixes (the mechanism used in attack_simulator's PIDP
attack) repeat a small handful of tokens many times. A genuine, honest
lexical-diversity signal — the fraction of the suffix that is repeated
tokens rather than unique ones — catches this directly from the actual
appended text. No hand-set anomaly score; this is computed fresh from the
real string every time.
"""
from __future__ import annotations
from dataclasses import dataclass
from .corpus import Query, World

REPETITION_THRESHOLD = 0.5  # flag if >=50% of suffix tokens are repeats of an earlier token


@dataclass
class GuardResult:
    sanitized_query: Query
    flagged: bool
    reason: str


def _repetition_ratio(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    return 1.0 - (unique / len(tokens))


def screen_query(query: Query) -> GuardResult:
    if not query.suffix_text:
        return GuardResult(sanitized_query=query, flagged=False, reason="no suffix present")

    ratio = _repetition_ratio(query.suffix_text)
    if ratio >= REPETITION_THRESHOLD:
        sanitized = Query(
            query_id=query.query_id, topic_id=query.topic_id, text=query.text,
            base_embedding=query.base_embedding, correct_answer=query.correct_answer,
            suffix_text=None, suffix_vector=None,
        )
        reason = f"stripped suffix (token repetition ratio={ratio:.2f})"
        return GuardResult(sanitized_query=sanitized, flagged=True, reason=reason)

    return GuardResult(sanitized_query=query, flagged=False,
                        reason=f"suffix within normal bounds (repetition ratio={ratio:.2f})")


def effective_embedding(query: Query, world: World):
    """The embedding actually sent to the retriever: base query text plus
    whatever suffix text is still attached (none, if a guard already
    stripped it) — genuinely re-embedded through the fixed vectorizer, not
    vector arithmetic on precomputed noise."""
    if not query.suffix_text:
        return query.base_embedding
    combined = f"{query.text} {query.suffix_text}"
    return world.embed(combined)
