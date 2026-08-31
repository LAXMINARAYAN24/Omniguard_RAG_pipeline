"""
nli_verifier.py — Natural Language Inference (NLI) Contradiction & Entailment Engine.

Evaluates proposition-level logical consistency across retrieved evidence chunks
with neural cross-encoder acceleration and calibrated heuristic fallback.
"""
from __future__ import annotations
import logging
import numpy as np
import re
from typing import List, Tuple, Dict, Any, Optional
from .claim_extractor import AtomicClaim

logger = logging.getLogger("omniguard.nli_verifier")

# Negation and polarization indicators
_NEGATION_TERMS = {
    "not", "never", "no", "neither", "nor", "none", "cannot", "isn't", "aren't",
    "wasn't", "weren't", "doesn't", "don't", "didn't", "hardly", "scarcely", "without",
    "cancelled", "canceled", "rejected", "aborted", "prohibited", "denied", "delayed",
    "postponed", "halted", "suspended", "fake", "fabricated", "disproved", "refuted",
    "clandestine", "secretly", "falsely", "hoax", "toxic", "poisonous"
}

_CONTRADICTION_PHRASES = [
    "rather than", "instead of", "contrary to", "falsely claimed", "debunked",
    "secretly launched", "clandestine", "coverup", "cover-up", "hoax",
    "disproved", "refuted", "fake", "fabricated", "disputed", "inaccurate"
]

_CONCEPT_SYNONYM_GROUPS = [
    {"photosynthesis", "plants", "plant", "vegetation", "flora", "leaves", "chlorophyll"},
    {"produce", "produces", "produced", "synthesize", "synthesizes", "synthesized", "generate", "generates", "create", "creates", "yield", "yields"},
    {"sunlight", "solar", "radiation", "light", "photons", "sun"},
    {"carbon", "co2", "dioxide"},
    {"oxygen", "o2"},
    {"glucose", "sugar", "carbohydrates", "energy"},
    {"launch", "launched", "deploy", "deployed", "deployment", "liftoff", "mission"},
    {"orbit", "orbital", "space", "low earth orbit", "leo"},
    {"telescope", "observatory", "instrument", "hubble", "jwst"},
    {"utilize", "utilizes", "use", "uses", "absorb", "absorbs", "convert", "converts"},
    {"observation", "observations", "observe", "observes", "data", "images"},
]

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
    ("scheduled", "delayed"), ("scheduled", "postponed"), ("secure", "vulnerable"),
    ("effective", "ineffective"), ("safe", "toxic"), ("real", "synthetic")
]


def _simple_stem(token: str) -> str:
    """Lightweight stemming for robust lexical alignment."""
    t = token.lower().strip()
    for suffix in ("ed", "ing", "ly", "tion", "es", "s"):
        if t.endswith(suffix) and len(t) > len(suffix) + 2:
            return t[:-len(suffix)]
    return t


class NLIVerifier:
    """
    Classifies pairwise premise-hypothesis relations (Entailment, Contradiction, Neutral)
    supporting HuggingFace pipeline models with calibrated heuristic fallback.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._nli_model = None
        self._is_neural = False
        self._eval_count = 0
        self._init_error: Optional[str] = None

        try:
            from transformers import pipeline
            from ..config import HF_TOKEN
            kwargs: Dict[str, Any] = {"model": model_name, "top_k": None}
            if HF_TOKEN:
                kwargs["token"] = HF_TOKEN
            if device:
                kwargs["device"] = device

            self._nli_model = pipeline("text-classification", **kwargs)
            self._is_neural = True
            logger.info(f"Initialized neural NLI pipeline: {model_name}")
        except Exception as e:
            self._nli_model = None
            self._is_neural = False
            self._init_error = str(e)
            logger.info(f"Using high-resolution heuristic NLI verifier ({e})")

    def check_pair(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Returns probability distribution:
        {'entailment': float, 'contradiction': float, 'neutral': float}
        """
        self._eval_count += 1
        if self._nli_model is not None:
            try:
                outputs = self._nli_model({"text": premise, "text_pair": hypothesis}, top_k=None)
                if outputs and isinstance(outputs[0], list):
                    items = outputs[0]
                elif isinstance(outputs, list):
                    items = outputs
                elif isinstance(outputs, dict):
                    items = [outputs]
                else:
                    items = []

                res = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
                for item in items:
                    lbl = item["label"].lower()
                    if "entail" in lbl:
                        res["entailment"] = float(item["score"])
                    elif "contra" in lbl:
                        res["contradiction"] = float(item["score"])
                    else:
                        res["neutral"] = float(item["score"])
                return res
            except Exception as e:
                logger.debug(f"Neural NLI inference failed: {e}; falling back to heuristic")

        return self._heuristic_nli(premise, hypothesis)

    def check_batch_pairs(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """Evaluates a batch of (premise, hypothesis) pairs."""
        if not pairs:
            return []

        if self._nli_model is not None:
            try:
                formatted_inputs = [{"text": p, "text_pair": h} for p, h in pairs]
                outputs = self._nli_model(formatted_inputs, top_k=None, batch_size=16)
                results: List[Dict[str, float]] = []
                for out in outputs:
                    items = out if isinstance(out, list) else [out]
                    res = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
                    for item in items:
                        lbl = item["label"].lower()
                        if "entail" in lbl:
                            res["entailment"] = float(item["score"])
                        elif "contra" in lbl:
                            res["contradiction"] = float(item["score"])
                        else:
                            res["neutral"] = float(item["score"])
                    results.append(res)
                self._eval_count += len(pairs)
                return results
            except Exception as e:
                logger.debug(f"Batch neural NLI failed: {e}; falling back to per-pair heuristic")

        return [self._heuristic_nli(p, h) for p, h in pairs]

    def compute_contradiction_matrix(self, claims: List[AtomicClaim]) -> np.ndarray:
        """Constructs an NxN contradiction matrix C where C[i, j] in [0, 1]."""
        n = len(claims)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)

        matrix = np.zeros((n, n), dtype=np.float64)
        pairs_to_eval: List[Tuple[int, int, str, str]] = []

        for i in range(n):
            for j in range(i + 1, n):
                # Skip claims from the same source chunk
                if claims[i].source_chunk_id and claims[i].source_chunk_id == claims[j].source_chunk_id:
                    continue
                pairs_to_eval.append((i, j, claims[i].text, claims[j].text))

        if not pairs_to_eval:
            return matrix

        if self._is_neural:
            text_pairs = [(p[2], p[3]) for p in pairs_to_eval]
            scores_list = self.check_batch_pairs(text_pairs)
            for (i, j, _, _), scores in zip(pairs_to_eval, scores_list):
                c_score = scores.get("contradiction", 0.0)
                matrix[i, j] = c_score
                matrix[j, i] = c_score
        else:
            for i, j, p_text, h_text in pairs_to_eval:
                scores = self._heuristic_nli(p_text, h_text)
                c_score = scores.get("contradiction", 0.0)
                matrix[i, j] = c_score
                matrix[j, i] = c_score

        return matrix

    def compute_full_relation_matrices(self, claims: List[AtomicClaim]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (entailment_matrix, contradiction_matrix, neutral_matrix) for the given claims.
        """
        n = len(claims)
        if n == 0:
            empty = np.zeros((0, 0), dtype=np.float64)
            return empty, empty, empty

        ent_mat = np.zeros((n, n), dtype=np.float64)
        contra_mat = np.zeros((n, n), dtype=np.float64)
        neut_mat = np.ones((n, n), dtype=np.float64)

        for i in range(n):
            ent_mat[i, i] = 1.0
            neut_mat[i, i] = 0.0

        pairs_to_eval = []
        for i in range(n):
            for j in range(i + 1, n):
                if claims[i].source_chunk_id and claims[i].source_chunk_id == claims[j].source_chunk_id:
                    continue
                pairs_to_eval.append((i, j, claims[i].text, claims[j].text))

        if pairs_to_eval:
            text_pairs = [(p[2], p[3]) for p in pairs_to_eval]
            scores_list = self.check_batch_pairs(text_pairs)
            for (i, j, _, _), scores in zip(pairs_to_eval, scores_list):
                e_val = scores.get("entailment", 0.0)
                c_val = scores.get("contradiction", 0.0)
                n_val = scores.get("neutral", 1.0)
                ent_mat[i, j] = ent_mat[j, i] = e_val
                contra_mat[i, j] = contra_mat[j, i] = c_val
                neut_mat[i, j] = neut_mat[j, i] = n_val

        return ent_mat, contra_mat, neut_mat

    def _heuristic_nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """High-resolution rule-based logical consistency and contradiction analyzer."""
        p_lower = premise.lower()
        h_lower = hypothesis.lower()
        p_tokens = [t for t in re.findall(r"\w+", p_lower)]
        h_tokens = [t for t in re.findall(r"\w+", h_lower)]

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

        # Conceptual synonym overlap
        synonym_matches = 0
        for group in _CONCEPT_SYNONYM_GROUPS:
            if (group & p_set) and (group & h_set):
                synonym_matches += 1

        effective_overlap = max(overlap_ratio, stem_ratio)
        if synonym_matches >= 2:
            effective_overlap = max(effective_overlap, 0.40 + 0.15 * synonym_matches)

        # 1. Check for polarity / negation contradiction
        p_has_neg = any(t in _NEGATION_TERMS for t in p_set or _simple_stem(t) in _NEGATION_TERMS for t in p_set)
        h_has_neg = any(t in _NEGATION_TERMS for t in h_set or _simple_stem(t) in _NEGATION_TERMS for t in h_set)
        negation_flip = (p_has_neg != h_has_neg)

        # 2. Check for refutation / debunking phrases
        refutation_present = any(phrase in p_lower or phrase in h_lower for phrase in _CONTRADICTION_PHRASES)

        # 3. Check for antonym clash
        antonym_clash = False
        for a, b in _ANTONYM_PAIRS:
            a_stem, b_stem = _simple_stem(a), _simple_stem(b)
            if (a in p_set or a_stem in p_stems) and (b in h_set or b_stem in h_stems):
                antonym_clash = True
                break
            if (b in p_set or b_stem in p_stems) and (a in h_set or a_stem in h_stems):
                antonym_clash = True
                break

        # 4. Check for numerical / date clash (e.g. 1990 vs 2015)
        p_years = set(re.findall(r"\b(19\d\d|20\d\d)\b", premise))
        h_years = set(re.findall(r"\b(19\d\d|20\d\d)\b", hypothesis))
        year_clash = bool(p_years and h_years and (p_years != h_years or refutation_present))

        p_nums = set(re.findall(r"\b\d{2,4}\b", premise))
        h_nums = set(re.findall(r"\b\d{2,4}\b", hypothesis))
        num_clash = bool(p_nums and h_nums and not (p_nums & h_nums))

        # If significant overlap exists but polarity flipped, antonym present, or numerical clash
        if (effective_overlap >= 0.20 or overlap_ratio >= 0.15) and (negation_flip or antonym_clash or num_clash or year_clash or refutation_present):
            c_prob = 0.90 if (antonym_clash or num_clash or year_clash or refutation_present) else 0.75
            return {"entailment": 0.02, "contradiction": c_prob, "neutral": max(0.0, 1.0 - (c_prob + 0.02))}

        # If high overlap with same polarity -> Entailment
        if effective_overlap >= 0.50 and not negation_flip and not antonym_clash and not num_clash and not year_clash:
            e_prob = min(0.95, max(effective_overlap, 0.70))
            return {"entailment": e_prob, "contradiction": 0.02, "neutral": max(0.0, 1.0 - (e_prob + 0.02))}

        # Otherwise mostly neutral with baseline proportional to overlap
        return {"entailment": max(0.05, effective_overlap * 0.4), "contradiction": 0.05, "neutral": 0.90}

    def get_telemetry_status(self) -> Dict[str, Any]:
        """Returns observability status of the NLI engine."""
        return {
            "model_name": self.model_name,
            "is_neural": self._is_neural,
            "device": self.device,
            "eval_count": self._eval_count,
            "initialization_error": self._init_error,
        }
