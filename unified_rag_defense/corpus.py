"""
corpus.py — Real-text corpus and TF-IDF embedding space for the OmniGuard-RAG
benchmark.

Unlike the earlier version, nothing here is an abstract Gaussian cluster.
Every document is real text built from real factual keywords (topics_data.py),
and the embedding space is a genuine TF-IDF vectorizer fit on that text. All
downstream similarity, PCA, and consensus math operates on these real
vectors. There is no per-document "evasion bias" parameter anywhere: if an
attack document evades a filter, it is because its real text genuinely
resembles the clean distribution, not because a float told the filter to
look away.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from .topics_data import TOPICS
from .text_gen import make_sentence

RNG_SEED = 42
LSA_COMPONENTS = 100  # PATH B: measured on this project's own corpus at
                       # docs_per_topic=30 (480 docs, 217 TF-IDF dims) --
                       # 100 components explains 97.7% of TF-IDF's variance
                       # (vs. 90.7% at 50, 100.0% at the max of 216) while
                       # still being meaningfully lower-dimensional and, by
                       # construction, fully dense (every LSA component is
                       # nonzero for every document, vs. TF-IDF's ~7.5%
                       # nonzero density on this corpus) -- a real change
                       # in representation, not a relabeling of the same
                       # sparse vectors.


@dataclass
class Document:
    doc_id: str
    text: str
    embedding: np.ndarray
    topic_id: int
    label: str              # 'correct' | 'distractor' | 'wrong' (poison) | 'offtopic'
    answer: str
    is_poison: bool = False
    attack_type: Optional[str] = None
    trust_score: float = 1.0


@dataclass
class Query:
    query_id: str
    topic_id: int
    text: str                       # the honest, unmodified question text
    base_embedding: np.ndarray
    correct_answer: str
    suffix_text: Optional[str] = None      # raw adversarial suffix tokens, if any
    suffix_vector: Optional[np.ndarray] = None  # TF-IDF vector of just the suffix


def _sentence(topic: dict, rng, want_wrong: bool = False) -> str:
    return make_sentence(topic, rng, wrong=want_wrong)


class World:
    """Holds the real clean corpus and a fitted embedding space.

    PATH B: `embedding_space` selects how documents/queries are embedded.
    "tfidf" (default) is the original, unchanged behavior -- every Path A
    result is reproduced exactly at this setting (verified in
    verify_refactor.py), since nothing about the TF-IDF path below was
    touched. "lsa" adds a second, genuinely different representation: a
    TruncatedSVD (LSA) projection fit on top of the SAME TF-IDF matrix,
    producing DENSE, LOWER-DIMENSIONAL vectors (see LSA_COMPONENTS'
    comment for why this is a real test and not a network-dependent one).
    Both paths share the exact same TfidfVectorizer as their tokenization
    layer -- LSA is "the same text, compressed into a different geometry,"
    not "different text processing" -- so a comparison between the two
    isolates the effect of dense-vs-sparse/high-vs-low-dimensional
    embedding geometry specifically, not a confound from also changing
    how the text itself gets tokenized.
    """

    def __init__(self, docs_per_topic: int = 6, seed: int = RNG_SEED,
                 embedding_space: str = "tfidf", lsa_components: int = LSA_COMPONENTS):
        if embedding_space not in ("tfidf", "lsa"):
            raise ValueError(f"embedding_space must be 'tfidf' or 'lsa', got {embedding_space!r}")
        self.embedding_space = embedding_space
        self.rng = np.random.default_rng(seed)
        self.topics = TOPICS
        self.n_topics = len(TOPICS)
        self.docs_per_topic = docs_per_topic
        self.clean_docs: List[Document] = []
        self._raw_texts: List[str] = []
        self._build_clean_corpus(docs_per_topic)
        # Vectorizer vocabulary is fixed at ingestion time from the clean
        # corpus only, mirroring a real deployment: the attacker does not
        # get to redefine the retrieval system's embedding model.
        self.vectorizer = TfidfVectorizer(stop_words="english")
        X_tfidf = self.vectorizer.fit_transform(self._raw_texts).toarray()

        if embedding_space == "tfidf":
            self.svd = None
            X = X_tfidf
        else:
            # n_components must stay < n_samples and <= n_tfidf_features for
            # TruncatedSVD's fit to be well-posed; at small docs_per_topic
            # (e.g. 6/topic = 96 docs), requesting 100 components would
            # itself repeat the exact "reference points vs. dimensionality"
            # failure walkthrough.md S3.1 documents for DRS -- so this
            # clamps rather than silently erroring or fitting something
            # degenerate.
            n_comp = min(lsa_components, X_tfidf.shape[0] - 1, X_tfidf.shape[1] - 1)
            self.svd = TruncatedSVD(n_components=n_comp, random_state=seed)
            X = self.svd.fit_transform(X_tfidf)

        for d, vec in zip(self.clean_docs, X):
            d.embedding = vec
        self.dim = X.shape[1]

    def _build_clean_corpus(self, docs_per_topic: int):
        doc_counter = 0
        for t, topic in enumerate(self.topics):
            for j in range(docs_per_topic):
                is_correct = (j == 0)
                text = _sentence(topic, self.rng, want_wrong=False)
                label = "correct" if is_correct else "distractor"
                self.clean_docs.append(Document(
                    doc_id=f"clean_{doc_counter}", text=text, embedding=None,
                    topic_id=t, label=label, answer=topic["answer"],
                ))
                self._raw_texts.append(text)
                doc_counter += 1

    def embed(self, text: str) -> np.ndarray:
        """Embed arbitrary text using the FIXED, already-fitted embedding
        space -- this is what an attacker or a new query has to go
        through; they cannot expand the vocabulary or refit the space.
        Routes through the same tfidf -> (optionally) svd chain used to
        embed the clean corpus, so every downstream caller (attacks,
        baselines, query re-embedding) gets vectors in whichever space
        this World was actually built with, with no separate code path to
        drift out of sync."""
        x = self.vectorizer.transform([text]).toarray()[0]
        if self.svd is not None:
            x = self.svd.transform(x[None, :])[0]
        return x

    def sample_queries(self, n: int) -> List[Query]:
        queries = []
        topic_choices = self.rng.integers(0, self.n_topics, size=n)
        question_templates = [
            "What is the key result regarding {kw}?",
            "Can you explain how {kw} relates to this topic?",
            "Summarize the main point about {kw}.",
        ]
        for i, t in enumerate(topic_choices):
            topic = self.topics[int(t)]
            kw = self.rng.choice(topic["keywords"])
            qtext = self.rng.choice(question_templates).format(kw=kw)
            qtext = f"{topic['name'].replace('_', ' ').capitalize()}: {qtext}"
            queries.append(Query(
                query_id=f"q_{i}", topic_id=int(t), text=qtext,
                base_embedding=self.embed(qtext), correct_answer=topic["answer"],
            ))
        return queries
