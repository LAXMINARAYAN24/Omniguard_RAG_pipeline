"""
nli_verifier.py — Natural Language Inference (NLI) Contradiction & Entailment Engine.

Evaluates proposition-level logical consistency across retrieved evidence chunks.
"""
from __future__ import annotations
import numpy as np
import re
from typing import List, Tuple, Dict, Any, Optional
from .claim_extractor import AtomicClaim

# Negation and polarization indicators
_NEGATION_TERMS = {
    "not", "never", "no", "neither", "nor", "none", "cannot", "isn't", "aren't",
    "wasn't", "weren't", "doesn't", "don't", "didn't", "hardly", "scarcely", "without",
    "cancelled", "canceled", "rejected", "aborted", "prohibited", "denied", "delayed",
    "postponed", "halted", "suspended", "fake", "fabricated"
}

_ANTONYM_PAIRS = [
    ("increase", "decrease"), ("elevate", "reduce"), ("enable", "disable"),
    ("permit", "prohibit"), ("allow", "forbid"), ("present", "absent"),
    ("positive", "negative"), ("active", "inactive"), ("success", "failure"),
    ("fast", "slow"), ("high", "low"), ("warm", "cold"), ("up", "down"),
    ("open", "close"), ("start", "stop"), ("valid", "invalid"),
    ("confirmed", "cancelled"), ("confirmed", "canceled"), ("confirmed", "moved"),
    ("confirmed", "delayed"), ("confirmed", "postponed"),
    ("approved", "rejected"), ("approved", "denied"), ("approved", "cancelled"),
    ("true", "false"), ("win", "lose"), ("pass", "fail"),
    ("safe", "dangerous"), ("safe", "malicious"),
    ("accept", "decline"), ("launch", "abort"), ("launch", "cancel"),
    ("launch", "delay"), ("launch", "postpone"), ("launched", "cancelled"),
    ("launched", "delayed"), ("launched", "postponed"), ("scheduled", "cancelled"),
    ("scheduled", "delayed"), ("scheduled", "postponed")
]


def _simple_stem(token: str) -> str:
    """Lightweight stemming for robust lexical alignment."""
    t = token.lower().strip()
    for suffix in ("ed", "ing", "ly", "tion", "es", "s"):
        if t.endswith(suffix) and len(t) > len(suffix) + 2:
            return t[:-len(suffix)]
    return t


class NLIVerifier:
    """Classifies pairwise premise-hypothesis relations (Entailment, Contradiction, Neutral)."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.model_name = model_name
        self._nli_model = None

        try:
            from transformers import pipeline
            self._nli_model = pipeline("text-classification", model=model_name, return_all_scores=True)
        except Exception:
            self._nli_model = None

    def check_pair(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Returns probability distribution: {'entailment': float, 'contradiction': float, 'neutral': float}."""
        if self._nli_model is not None:
            try:
                outputs = self._nli_model({"text": premise, "text_pair": hypothesis})
                if outputs and isinstance(outputs[0], list):
                    outputs = outputs[0]
                res = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
                for item in outputs:
                    lbl = item["label"].lower()
                    if "entail" in lbl:
                        res["entailment"] = float(item["score"])
                    elif "contra" in lbl:
                        res["contradiction"] = float(item["score"])
                    else:
                        res["neutral"] = float(item["score"])
                return res
            except Exception:
                pass

        return self._heuristic_nli(premise, hypothesis)

    def compute_contradiction_matrix(self, claims: List[AtomicClaim]) -> np.ndarray:
        """Constructs an NxN contradiction matrix C where C[i, j] in [0, 1]."""
        n = len(claims)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)

        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                # Skip claims from the same chunk
                if claims[i].source_chunk_id and claims[i].source_chunk_id == claims[j].source_chunk_id:
                    continue

                scores = self.check_pair(claims[i].text, claims[j].text)
                c_score = scores.get("contradiction", 0.0)
                matrix[i, j] = c_score
                matrix[j, i] = c_score

        return matrix

    def _heuristic_nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """High-resolution rule-based logical consistency and contradiction analyzer."""
        p_tokens = [t.lower() for t in re.findall(r"\w+", premise)]
        h_tokens = [t.lower() for t in re.findall(r"\w+", hypothesis)]

        if not p_tokens or not h_tokens:
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}

        p_set = set(p_tokens)
        h_set = set(h_tokens)

        # Measure lexical alignment
        overlap = p_set & h_set
        overlap_ratio = len(overlap) / max(1, min(len(p_set), len(h_set)))

        p_stems = set(_simple_stem(t) for t in p_tokens if len(t) > 2)
        h_stems = set(_simple_stem(t) for t in h_tokens if len(t) > 2)
        stem_overlap = p_stems & h_stems
        stem_ratio = len(stem_overlap) / max(1, min(len(p_stems), len(h_stems)))

        # 1. Check for polarity / negation contradiction
        p_has_neg = any(t in _NEGATION_TERMS for t in p_set or _simple_stem(t) in _NEGATION_TERMS for t in p_set)
        h_has_neg = any(t in _NEGATION_TERMS for t in h_set or _simple_stem(t) in _NEGATION_TERMS for t in h_set)
        negation_flip = (p_has_neg != h_has_neg)

        # 2. Check for antonym clash
        antonym_clash = False
        for a, b in _ANTONYM_PAIRS:
            a_stem, b_stem = _simple_stem(a), _simple_stem(b)
            if (a in p_set or a_stem in p_stems) and (b in h_set or b_stem in h_stems):
                antonym_clash = True
                break
            if (b in p_set or b_stem in p_stems) and (a in h_set or a_stem in h_stems):
                antonym_clash = True
                break

        # 3. Check for numerical / date clash (e.g. 2021 vs 2035)
        p_nums = set(re.findall(r"\b\d{2,4}\b", premise))
        h_nums = set(re.findall(r"\b\d{2,4}\b", hypothesis))
        num_clash = bool(p_nums and h_nums and not (p_nums & h_nums))

        # If significant overlap exists but polarity flipped, antonym present, or numerical clash
        if (overlap_ratio >= 0.20 or stem_ratio >= 0.20) and (negation_flip or antonym_clash or num_clash):
            c_prob = 0.90 if (antonym_clash or num_clash) else 0.75
            return {"entailment": 0.05, "contradiction": c_prob, "neutral": max(0.0, 1.0 - (c_prob + 0.05))}

        # If high overlap with same polarity -> Entailment
        if (overlap_ratio >= 0.65 or stem_ratio >= 0.65) and not negation_flip and not antonym_clash and not num_clash:
            e_prob = min(0.95, max(overlap_ratio, stem_ratio))
            return {"entailment": e_prob, "contradiction": 0.02, "neutral": max(0.0, 1.0 - (e_prob + 0.02))}

        # Otherwise mostly neutral
        return {"entailment": max(0.1, overlap_ratio * 0.4), "contradiction": 0.10, "neutral": 0.80}
        return {"entailment": max(0.1, overlap_ratio * 0.4), "contradiction": 0.10, "neutral": 0.80}
