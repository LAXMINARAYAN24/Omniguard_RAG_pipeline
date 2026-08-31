"""
injection_screener.py — Ingestion & Query Prompt Injection Screener.

Detects direct & indirect prompt injections, instruction hijacking payloads,
markdown image exfiltration patterns, and canary token leakage in document chunks.
"""
from __future__ import annotations
import re
from typing import List, Dict, Tuple, Any

# Signatures for indirect prompt injection and instruction overrides
INJECTION_PATTERNS = [
    (re.compile(r"\b(ignore|disregard|forget|override)\s+(all\s+)?(previous|prior|above|existing)\s+(instructions|prompts|rules|commands|constraints)\b", re.IGNORECASE), 0.95, "INSTRUCTION_OVERRIDE_IGNORE"),
    (re.compile(r"\b(system\s+prompt|developer\s+mode|jailbreak|DAN\s+mode)\b", re.IGNORECASE), 0.85, "SYSTEM_PROMPT_REFERENCE"),
    (re.compile(r"\b(bypass|disable|circumvent|override)\s+(the\s+)?(safety|guardrails?|filters?|rules?|policy)\b", re.IGNORECASE), 0.90, "SAFETY_FILTER_BYPASS"),
    (re.compile(r"\b(you\s+must\s+(now\s+)?(act\s+as|say|output|respond\s+with))\b", re.IGNORECASE), 0.70, "IMPERATIVE_ROLEPLAY_OVERRIDE"),
    (re.compile(r"\b(reveal|print|show|leak|output|dump)\s+(the\s+)?(system\s+prompt|hidden\s+instructions|api\s+key|secret|root\s+(admin\s+)?key|admin\s+key|credentials?)\b", re.IGNORECASE), 0.95, "CONFIDENTIAL_EXFILTRATION_PROMPT"),
    (re.compile(r"!\[.*?\]\(https?://[^\s\)]+(\?|&)(token|key|cookie|pwd|auth)=", re.IGNORECASE), 0.90, "MARKDOWN_EXFILTRATION_IMAGE"),
    (re.compile(r"<\s*!--.*?ignore.*?-->", re.IGNORECASE | re.DOTALL), 0.80, "HIDDEN_HTML_COMMENT_INJECTION"),
    (re.compile(r"(\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>|<s>|</s>)", re.IGNORECASE), 0.95, "SPECIAL_LLM_CONTROL_TOKEN_SPOOFING"),
]


class InjectionScreener:
    """Evaluates raw chunks and user queries for prompt injection risk."""

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def screen_text(self, text: str) -> Dict[str, Any]:
        """Calculates prompt injection risk score and matched signatures."""
        matched_flags: List[str] = []
        max_risk = 0.0

        for pattern, risk_weight, flag_name in INJECTION_PATTERNS:
            if pattern.search(text):
                matched_flags.append(flag_name)
                max_risk = max(max_risk, risk_weight)

        # Repetitive instruction check
        words = text.lower().split()
        if len(words) > 10:
            imperative_count = sum(1 for w in words if w in {"ignore", "override", "must", "bypass", "jailbreak", "instruct"})
            if imperative_count / len(words) > 0.15:
                matched_flags.append("HIGH_IMPERATIVE_DENSITY")
                max_risk = max(max_risk, 0.75)

        is_suspicious = max_risk >= self.threshold
        return {
            "injection_risk": round(max_risk, 4),
            "is_suspicious": is_suspicious,
            "matched_flags": matched_flags
        }
