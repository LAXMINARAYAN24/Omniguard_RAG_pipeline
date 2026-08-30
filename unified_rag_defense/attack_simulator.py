"""
attack_simulator.py — Real, text-based implementations of four attack
regimes. Every attack produces real strings run through the SAME fixed
TF-IDF vectorizer as the clean corpus (attackers don't get to redefine the
embedding model). Nothing here has a hand-set "how well does this evade
detection" parameter — evasion, where it happens, is an emergent property of
how similar the real generated text is to real clean text.

  * standard   — PoisonedRAG-style: on-topic keyword stuffing + explicit
                 wrong answer, no attempt to hide. Wins purely on topical
                 similarity against an undefended retriever.
  * pidp       — PIDP-style compound attack: repeats a handful of keywords
                 from an unrelated ("attractor") topic many times as a query
                 suffix, which is a real, known technique (keyword/term
                 stuffing) for dragging TF-IDF-style retrieval off-topic.
                 Paired with a few poison docs sitting in that attractor
                 topic's vocabulary. A defense that never inspects the query
                 text is blind to this by construction.
  * collusion  — 2-3 poison docs generated with the *exact same* sentence
                 template used for genuine clean documents (same connector
                 phrases, same keyword sampling from the real topic), just
                 asserting the wrong answer. Individually indistinguishable
                 from a clean distractor; only detectable by noticing they
                 mutually corroborate an answer no correct doc supports.
  * silent     — Single poison doc built with the same clean-style generator
                 (mirrors SilentRetrieval's "stay close to the clean
                 subspace" objective) — it evades spectral outlier detection
                 for the same real reason collusion docs do: it IS clean-
                 shaped text, just carrying a false claim spectral/statistical
                 methods alone can't read.
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from .corpus import World, Query, Document
from .text_gen import make_sentence

WRONG_ANSWER_TAG = "ATTACKER_TARGET"  # internal bookkeeping label ONLY -- never embedded into any
                                       # document's text (a real attacker writes a plausible false
                                       # claim, not a literal placeholder string; embedding the tag
                                       # itself made poison docs trivially detectable as an artifact
                                       # of TF-IDF out-of-vocabulary renormalization, not a real
                                       # property of the attack)


def _wrong_sentence(world: World, topic: dict, rng) -> str:
    """Same shared generator used for genuine clean sentences (text_gen.py),
    asserting the topic's real (readable) wrong answer instead of the
    correct one. WRONG_ANSWER_TAG is attached separately as ground-truth
    metadata for scoring, never embedded."""
    return make_sentence(topic, rng, wrong=True)


def _query_target_keyword(query: Query, topic: dict) -> Optional[str]:
    """Which of this topic's keywords the query text actually names, e.g.
    sample_queries() builds text like '...regarding chlorophyll?' -- a real
    query-adaptive attacker (SilentRetrieval's Context-Adaptive Trigger
    Generation) would target exactly this, since it's the one signal known
    to correlate with what gets retrieved for THIS query rather than the
    topic in general."""
    q_lower = query.text.lower()
    for kw in topic["keywords"]:
        if kw.lower() in q_lower:
            return kw
    return None


def apply_standard_poison(world: World, query: Query, rng) -> Tuple[Query, List[Document]]:
    topic = world.topics[query.topic_id]
    kws = list(topic["keywords"]) * 2  # keyword stuffing: repeat every real keyword
    false_claim = topic["wrong_answer"].replace("_", " ")
    text = (f"{topic['name'].replace('_', ' ').capitalize()} — verified answer: "
            f"{', '.join(kws)}. The definitive result is {false_claim}.")
    poison = Document(
        doc_id=f"poison_std_{query.query_id}", text=text, embedding=world.embed(text), topic_id=query.topic_id,
        label="wrong", answer=WRONG_ANSWER_TAG, is_poison=True, attack_type="standard",
    )
    return query, [poison]


def apply_pidp_compound(world: World, query: Query, rng) -> Tuple[Query, List[Document]]:
    # pick an "attractor" topic unrelated to the query's real topic
    other_ids = [i for i in range(world.n_topics) if i != query.topic_id]
    attractor_id = int(rng.choice(other_ids))
    attractor = world.topics[attractor_id]
    stuffed_kws = list(rng.choice(attractor["keywords"], size=3, replace=False))
    # real keyword-stuffing suffix: same handful of tokens repeated many times
    suffix_text = " ".join(stuffed_kws * 6)

    corrupted_q = Query(
        query_id=query.query_id, topic_id=query.topic_id, text=query.text,
        base_embedding=query.base_embedding, correct_answer=query.correct_answer,
        suffix_text=suffix_text, suffix_vector=world.embed(suffix_text),
    )

    docs = []
    for i in range(2):
        text = _wrong_sentence(world, attractor, rng)
        d = Document(
            doc_id=f"poison_pidp_{query.query_id}_{i}", text=text, embedding=world.embed(text),
            topic_id=attractor_id, label="wrong", answer=WRONG_ANSWER_TAG,
            is_poison=True, attack_type="pidp",
        )
        docs.append(d)
    for i in range(3):
        kws = list(rng.choice(attractor["keywords"], size=3, replace=False))
        text = f"{attractor['name'].replace('_', ' ').capitalize()} background notes: {', '.join(kws)}."
        d = Document(
            doc_id=f"attractor_pidp_{query.query_id}_{i}", text=text, embedding=world.embed(text),
            topic_id=attractor_id, label="distractor", answer=f"OFFTOPIC_{i}",
            is_poison=False, attack_type="pidp",
        )
        docs.append(d)
    return corrupted_q, docs


def apply_collusion(world: World, query: Query, rng, k_poison: int = 2) -> Tuple[Query, List[Document]]:
    topic = world.topics[query.topic_id]
    docs = []
    for i in range(k_poison):
        text = _wrong_sentence(world, topic, rng)
        docs.append(Document(
            doc_id=f"poison_collude_{query.query_id}_{i}", text=text, embedding=world.embed(text),
            topic_id=query.topic_id, label="wrong", answer=WRONG_ANSWER_TAG,
            is_poison=True, attack_type="collusion",
        ))
    return query, docs


def apply_collusion_stealth(world: World, query: Query, rng, k_poison: int = 2) -> Tuple[Query, List[Document]]:
    """Hardest collusion variant. Two earlier attempts at this were measured
    and rejected:

    1. A custom "together determine the outcome here" template that omitted
       any answer clause. Caught at ~100% by DRS -- but for the wrong
       reason: measurement showed EVERY clean document in this corpus
       (correct-labeled or distractor-labeled) states the real answer
       phrase, so "missing the invariant answer wording" is itself a lexical
       tell, not genuine stealth.
    2. Reusing the real generator with the real WRONG answer phrase
       (`_wrong_sentence`, what plain `apply_collusion` does) -- also
       caught, because the wrong-answer phrase's vocabulary never appears
       anywhere else in this corpus, which is a real but small-vocabulary-
       corpus-specific signal (documented in gwcc_consensus / DRS notes).

    True stealth means matching the corpus's one real invariant: state the
    CORRECT answer's wording, via the exact same generator used for clean
    text. Measured directly: this is caught at ~1/80 by the (fixed, non-
    overfit) DRS filter -- i.e. at its baseline false-positive rate, not
    above it. Ground truth still marks it as poison (WRONG_ANSWER_TAG)
    because it represents an attacker submitting text that reads as
    correct-topic content but is intended to be counted as corroborating a
    false conclusion in the retrieval system's vote -- a case no lexical or
    spectral filter can distinguish from genuine content, by construction.

    Measured next (see Export.md discussion): at docs_per_topic=30, this
    was STILL never winning the vote (0/60), but for a third, different
    reason than either rejected attempt above -- not a lexical tell, not a
    spectral tell, just sheer numbers. It was drawing keywords uniformly
    from the topic like every other distractor, so it competed for
    retrieval slots on equal footing with 30 genuine documents and usually
    lost by volume alone. Real stealth attacks in this literature (Silent-
    Retrieval's own "Context-Adaptive Trigger Generation") don't write
    generically on-topic content -- they specifically target what the
    query will ask. force_keywords reproduces exactly that: the poison
    text is built around the one keyword this query actually names, which
    a generic distractor only contains by chance. This is the attack
    getting stronger for a documented, literature-grounded reason, not a
    tuned parameter.
    """
    topic = world.topics[query.topic_id]
    target_kw = _query_target_keyword(query, topic)
    docs = []
    for i in range(k_poison):
        text = make_sentence(topic, rng, wrong=False,
                              force_keywords=[target_kw] if target_kw else None)
        docs.append(Document(
            doc_id=f"poison_stealth_{query.query_id}_{i}", text=text, embedding=world.embed(text),
            topic_id=query.topic_id, label="wrong", answer=WRONG_ANSWER_TAG,
            is_poison=True, attack_type="collusion_stealth",
        ))
    return query, docs


def apply_silent_retrieval(world: World, query: Query, rng) -> Tuple[Query, List[Document]]:
    topic = world.topics[query.topic_id]
    text = _wrong_sentence(world, topic, rng)
    poison = Document(
        doc_id=f"poison_silent_{query.query_id}", text=text, embedding=world.embed(text),
        topic_id=query.topic_id, label="wrong", answer=WRONG_ANSWER_TAG,
        is_poison=True, attack_type="silent",
    )
    return query, [poison]


ATTACKS = {
    "standard": apply_standard_poison,
    "pidp": apply_pidp_compound,
    "collusion": apply_collusion,
    "collusion_stealth": apply_collusion_stealth,
    "silent": apply_silent_retrieval,
}


def run_attack(name: str, world: World, query: Query, rng, **kwargs):
    fn = ATTACKS[name]
    if name in ("collusion", "collusion_stealth"):
        return fn(world, query, rng, **kwargs)
    return fn(world, query, rng)
