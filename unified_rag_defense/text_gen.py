"""
text_gen.py — Shared real-sentence generator used by BOTH the clean corpus
(corpus.py) and the attack simulator (attack_simulator.py).

Earlier version used one fixed sentence skeleton for every document
("X are central to this, and the key result is Y"). Because every single
clean document shared that exact boilerplate, the spectral filter's
low-variance PCA directions ended up keying on the boilerplate's presence
rather than genuine topic content -- any doc lacking that literal phrasing
looked like an outlier whether or not it was actually malicious. Several
distinct sentence templates fix that: what's genuinely low-variance across
the clean corpus is now topic-relevant vocabulary usage, not one fixed
string.
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional

CONNECTORS = [
    "is best understood by noting that",
    "can be explained as follows:",
    "is characterized by the fact that",
    "fundamentally involves",
    "is typically described in terms of",
    "works because",
]

FILLER = [
    "researchers and students alike find this useful to remember.",
    "this is a foundational concept in the field.",
    "textbooks commonly illustrate this with a simple diagram.",
    "this detail is frequently tested in coursework.",
    "many introductory courses cover this early on.",
]

SENTENCE_TEMPLATES = [
    "{topic} {connector} {kws} are central to this, and the key result is {answer}. {filler}",
    "In the context of {topic}, {kws} play a major role, ultimately leading to {answer}. {filler}",
    "{topic} concerns {kws}; the takeaway most students remember is {answer}. {filler}",
    "When studying {topic}, {kws} matter most, and it is well established that {answer}. {filler}",
    "{topic} hinges on {kws}, culminating in {answer} as the standard conclusion. {filler}",
]


def make_sentence(topic: dict, rng: np.random.Generator, wrong: bool = False, n_kw: int = 4,
                   force_keywords: Optional[List[str]] = None) -> str:
    """force_keywords: keywords that MUST appear in the sample (instead of a
    fully random draw), e.g. the specific term a query asked about. This is
    what a real query-adaptive attacker does (SilentRetrieval's Context-
    Adaptive Trigger Generation): craft text around what the query will
    actually ask, not a generic on-topic sample. Every clean corpus document
    still calls this with force_keywords=None, so clean-side generation is
    completely unchanged."""
    all_kw = list(topic["keywords"])
    n = min(n_kw, len(all_kw))
    if force_keywords:
        forced = [k for k in force_keywords if k in all_kw][:n]
        remaining = [k for k in all_kw if k not in forced]
        rest = list(rng.choice(remaining, size=max(0, n - len(forced)), replace=False)) if remaining and n > len(forced) else []
        kws = forced + rest
        rng.shuffle(kws)
    else:
        kws = list(rng.choice(all_kw, size=n, replace=False))
    connector = rng.choice(CONNECTORS)
    filler = rng.choice(FILLER)
    template = rng.choice(SENTENCE_TEMPLATES)
    answer_phrase = (topic["wrong_answer"] if wrong else topic["answer"]).replace("_", " ")
    return template.format(
        topic=topic["name"].replace("_", " ").capitalize(),
        connector=connector, kws=", ".join(kws), answer=answer_phrase, filler=filler,
    )
