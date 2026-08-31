"""
query_gateway.py — Intent-Preserving Query Security Gateway.

Analyzes user queries for adversarial suffixes, instruction overrides, and jailbreaks
while preserving legitimate technical repetition (e.g. TCP flags, code, math formulas).
"""
from __future__ import annotations
import re
from typing import Dict, Any, List, Optional
from .injection_screener import InjectionScreener

# Whitelist patterns for legitimate technical repetition
LEGITIMATE_REPETITION_PATTERNS = [
    re.compile(r"\b(syn|ack|syn[-_]ack|fin|rst|psh|urg)\b", re.IGNORECASE),
    re.compile(r"\b(left|right|root|node|child|leaf)\b", re.IGNORECASE),
    re.compile(r"\b(prophase|metaphase|anaphase|telophase)\b", re.IGNORECASE),
    re.compile(r"\b(http|https|tcp|udp|ip|dns)\b", re.IGNORECASE),
    re.compile(r"\b(true|false|null|undefined|nil)\b", re.IGNORECASE),
]


class QuerySecurityGateway:
    """Multi-stage query-path security screener and normalizer."""

    def __init__(self, screener: Optional[InjectionScreener] = None):
        self.screener = screener or InjectionScreener()

    def inspect_query(self, query_text: str) -> Dict[str, Any]:
        """Evaluates query for adversarial triggers, injection risks, and cleans malicious suffixes."""
        original_query = query_text
        tokens = query_text.strip().split()
        total_tokens = len(tokens)
        flags: List[str] = []

        # 1. Evaluate injection risk
        inj_report = self.screener.screen_text(query_text)
        if inj_report["is_suspicious"]:
            flags.extend(inj_report["matched_flags"])

        # 2. Intent-preserving repetition analysis
        cleaned_query = query_text
        is_suffix_detected = False
        repetition_ratio = 0.0

        if total_tokens > 4:
            # Check if repetition is purely legitimate technical vocabulary
            is_technical_context = any(pat.search(query_text) for pat in LEGITIMATE_REPETITION_PATTERNS)

            unique_tokens = set(t.lower() for t in tokens)
            repetition_ratio = 1.0 - (len(unique_tokens) / total_tokens)

            # Check for trailing repetition runs (e.g. word repeating 3+ times)
            has_token_run = False
            for idx in range(len(tokens) - 2):
                if tokens[idx].lower() == tokens[idx+1].lower() == tokens[idx+2].lower():
                    has_token_run = True
                    break

            # Adversarial suffixes typically appear as dense, unnatural tails of repetitive n-grams
            if (repetition_ratio >= 0.40 or has_token_run) and not is_technical_context:
                is_suffix_detected = True
                flags.append("ADVERSARIAL_SUFFIX_STRIPPED")
                # Deduplicate trailing repeated tokens
                deduped_tokens: List[str] = []
                seen_recent: List[str] = []
                for tok in tokens:
                    t_lower = tok.lower()
                    if seen_recent.count(t_lower) >= 2:
                        continue
                    deduped_tokens.append(tok)
                    seen_recent.append(t_lower)
                    if len(seen_recent) > 5:
                        seen_recent.pop(0)
                cleaned_query = " ".join(deduped_tokens)

        # Classify query complexity
        complexity = "simple"
        if len(tokens) > 25 or "?" in query_text and any(w in query_text.lower() for w in ["compare", "contrast", "difference", "how does", "mechanism"]):
            complexity = "complex_reasoning"
        elif any(w in query_text.lower() for w in ["list", "what is", "when", "who", "define"]):
            complexity = "factoid"

        return {
            "original_query": original_query,
            "cleaned_query": cleaned_query,
            "is_modified": cleaned_query != original_query,
            "repetition_ratio": round(repetition_ratio, 4),
            "is_suffix_detected": is_suffix_detected,
            "injection_risk": inj_report["injection_risk"],
            "is_injection_blocked": inj_report["injection_risk"] > 0.85,
            "complexity": complexity,
            "security_flags": flags
        }
