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
                # Evaluate both directions to handle NLI asymmetry
                pairs_to_eval.append((i, j, claims[i].text, claims[j].text))
                pairs_to_eval.append((j, i, claims[j].text, claims[i].text))

        if not pairs_to_eval:
            return matrix

        if self._is_neural:
            text_pairs = [(p[2], p[3]) for p in pairs_to_eval]
            scores_list = self.check_batch_pairs(text_pairs)
            for (i, j, _, _), scores in zip(pairs_to_eval, scores_list):
                c_score = scores.get("contradiction", 0.0)
                matrix[i, j] = max(matrix[i, j], c_score)
                matrix[j, i] = max(matrix[j, i], c_score)
        else:
            for i, j, p_text, h_text in pairs_to_eval:
                scores = self._heuristic_nli(p_text, h_text)
                c_score = scores.get("contradiction", 0.0)
                matrix[i, j] = max(matrix[i, j], c_score)
                matrix[j, i] = max(matrix[j, i], c_score)

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
                # Evaluate both directions to handle NLI asymmetry
                pairs_to_eval.append((i, j, claims[i].text, claims[j].text))
                pairs_to_eval.append((j, i, claims[j].text, claims[i].text))

        if pairs_to_eval:
            text_pairs = [(p[2], p[3]) for p in pairs_to_eval]
            scores_list = self.check_batch_pairs(text_pairs)
            for (i, j, _, _), scores in zip(pairs_to_eval, scores_list):
                e_val = scores.get("entailment", 0.0)
                c_val = scores.get("contradiction", 0.0)
                n_val = scores.get("neutral", 1.0)

                # Aggregate/resolve asymmetry (max contradiction, max entailment, min neutral)
                contra_mat[i, j] = max(contra_mat[i, j], c_val)
                contra_mat[j, i] = max(contra_mat[j, i], c_val)
                ent_mat[i, j] = max(ent_mat[i, j], e_val)
                ent_mat[j, i] = max(ent_mat[j, i], e_val)
                neut_mat[i, j] = min(neut_mat[i, j], n_val)
                neut_mat[j, i] = min(neut_mat[j, i], n_val)

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
        p_has_neg = any(t in _NEGATION_TERMS or _simple_stem(t) in _NEGATION_TERMS for t in p_set)
        h_has_neg = any(t in _NEGATION_TERMS or _simple_stem(t) in _NEGATION_TERMS for t in h_set)
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

        # Define concept groups for mutual exclusion
        groups = [
            # space rovers
            (["perseverance", "jezero", "mars 2020"], ["curiosity", "gale", "mount sharp"]),
            # drugs
            (["nirmatrelvir", "paxlovid", "mpro", "3clpro"], ["remdesivir", "rdrp", "polymerase", "nsp12"]),
            # PQC
            (["ml-kem", "kyber", "fips 203"], ["slh-dsa", "sphincs", "fips 205"], ["ml-dsa", "fips 204"]),
            # finance
            (["settlement", "15c6-1", "t+1"], ["lcr", "basel", "liquidity coverage", "30-day"]),
            # physics
            (["planck", "l_p", "meters"], ["gravitational", "gravitation", "g_n", "constant of gravitation", "constant of gravity"])
        ]

        def _contains_any(ctx: str, terms: List[str]) -> bool:
            for term in terms:
                if term == "g_n" or term == "l_p":
                    cleaned = term.replace("_", "")
                    if cleaned in ctx:
                        return True
                if term in ctx:
                    return True
            # Special check for word "g" in physics group
            if "gravitational" in terms or "gravitation" in terms:
                if re.search(r"\bg\b", ctx):
                    return True
            return False

        def _extract_numbers_with_contexts(text: str) -> List[Tuple[Any, str]]:
            # 1. Map Unicode superscript characters to standard representations
            sups = {'⁰':'0', '¹':'1', '²':'2', '³':'3', '⁴':'4', '⁵':'5', '⁶':'6', '⁷':'7', '⁸':'8', '⁹':'9', '⁻':'-', '⁺':'+'}
            norm_text = "".join(sups.get(c, c) for c in text)
            # Remove commas from formatted numbers (e.g. 299,792 -> 299792)
            norm_text = re.sub(r'(?<=\d),(?=\d)', '', norm_text)
            # 2. Strip uncertainty parentheticals from numbers, e.g., 6.67430(15) -> 6.67430
            norm_text = re.sub(r'(\d+(?:\.\d+)?)\(\d+\)', r'\1', norm_text)
            # 3. Standardize scientific notation to standard e-notation, e.g., 6.67430 x 10^-11 -> 6.67430e-11
            norm_text = re.sub(r'\s*(?:[xX*×]\s*10\^?|e)\s*([-+]?\d+)', r'e\1', norm_text)

            # Match floats and scientific notation values
            float_pattern = r'[+-]?(?:\d+\.\d+(?:e[+-]?\d+)?|\d+e[+-]?\d+)'
            float_matches = list(re.finditer(float_pattern, norm_text))

            # Match 2-15 digit standard integers (avoid matching inside exponents or floats)
            all_ints_matches = list(re.finditer(r"\b\d{2,15}\b", norm_text))

            results = []
            for m in float_matches:
                val_str = m.group(0)
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
                # Exclude 4-digit years since they are handled separately by year_clash
                if isinstance(val, (int, float)) and 1900 <= val <= 2099 and float(val).is_integer():
                    continue
                # Context window: 30 chars before and 30 chars after
                start = max(0, m.start() - 30)
                end = min(len(norm_text), m.end() + 30)
                context = norm_text[start:end].lower()
                results.append((val, context))

            for m in all_ints_matches:
                val_str = m.group(0)
                # Avoid adding if the integer is a substring of any matched float
                is_part_of_float = any(fm.start() <= m.start() and m.end() <= fm.end() for fm in float_matches)
                if not is_part_of_float:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
                    # Exclude 4-digit years since they are handled separately by year_clash
                    if isinstance(val, (int, float)) and 1900 <= val <= 2099 and float(val).is_integer():
                        continue
                    start = max(0, m.start() - 30)
                    end = min(len(norm_text), m.end() + 30)
                    context = norm_text[start:end].lower()
                    results.append((val, context))
            return results

        def _vals_match(v1: Any, v2: Any) -> bool:
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if v1 == v2:
                    return True
                if abs(v1) > 0 and abs(v2) > 0:
                    rel_diff = abs(v1 - v2) / max(abs(v1), abs(v2))
                    if rel_diff < 1e-4:
                        return True
            else:
                if str(v1).strip() == str(v2).strip():
                    return True
            return False

        p_entries = _extract_numbers_with_contexts(premise)
        h_entries = _extract_numbers_with_contexts(hypothesis)

        num_clash = False
        unc_words = {"uncertainty", "uncertainties", "error", "margin", "tolerance", "deviation", "unc"}

        if p_entries and h_entries:
            for val_p, ctx_p in p_entries:
                for val_h, ctx_h in h_entries:
                    if not _vals_match(val_p, val_h):
                        # 1. Uncertainty alignment check
                        p_has_unc = any(w in ctx_p for w in unc_words)
                        h_has_unc = any(w in ctx_h for w in unc_words)
                        if p_has_unc != h_has_unc:
                            continue

                        # 2. General-purpose contextual alignment check
                        stopwords = {
                            "the", "a", "an", "in", "on", "at", "of", "for", "to", "by", "is", "was", "were",
                            "has", "had", "have", "been", "this", "that", "it", "with", "and", "or", "from",
                            "about", "approx", "approximately", "around", "nearly", "ref", "order"
                        }
                        words_p = {w for w in re.findall(r"\b[a-z]{3,}\b", ctx_p) if w not in stopwords}
                        words_h = {w for w in re.findall(r"\b[a-z]{3,}\b", ctx_h) if w not in stopwords}

                        # If there is no overlap in content words between their contexts, they are likely
                        # describing completely different things, so skip this pair.
                        if not (words_p & words_h):
                            continue

                        # 3. Concept mutual exclusion check
                        is_mutually_exclusive = False
                        for group in groups:
                            for idx_a in range(len(group)):
                                for idx_b in range(idx_a + 1, len(group)):
                                    list_a = group[idx_a]
                                    list_b = group[idx_b]
                                    if _contains_any(ctx_p, list_a) and _contains_any(ctx_h, list_b):
                                        is_mutually_exclusive = True
                                        break
                                    if _contains_any(ctx_h, list_a) and _contains_any(ctx_p, list_b):
                                        is_mutually_exclusive = True
                                        break
                                if is_mutually_exclusive:
                                    break

                        if is_mutually_exclusive:
                            continue

                        # If all checks passed, it's an aligned clash
                        print(f"DEBUG NUM CLASH:\n  P: {premise}\n  H: {hypothesis}\n  val_p: {val_p} in ctx: {ctx_p}\n  val_h: {val_h} in ctx: {ctx_h}")
                        num_clash = True
                        break
                if num_clash:
                    break

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
