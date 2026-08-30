"""
gwcc_consensus.py — Ring 3: Group-Wise Counterfactual Consensus (GWCC).

BUGFIX (this session): the previous implementation took a "majority vote
across counterfactual subsets" -- treating each of the 1 full-set + k
singleton-removal + sampled pair-removal evaluations as one equal-weight
ballot, then picking whichever answer won the MOST of those ballots. That
sounds like it generalizes RAGuard/ZKIP's leave-one-out, but it doesn't:
diagnosed directly (see run_gwcc_diagnostic.py), when k'=3 colluding poison
docs already form a plurality in the full top-5 set, only the *few* subsets
that happen to remove enough of them flip away from the poison answer (e.g.
4 of 15 in a measured case) -- so the poison answer wins the aggregate vote
almost every time. Diagnostic confirmed 0/653 cases where this ever diverged
from a plain single-pass vote, across k_poison in [3, 5, 8, 12] -- Ring 3 was
computing extra calls for zero effect.

The fix restores the actual LOO *logic* RAGuard/ZKIP uses -- "does removing
this candidate change the answer? If so, exclude it and recompute" -- and
genuinely generalizes it to pairs, rather than voting on the candidate
removals as if they were independent evidence:

  1. Singleton pass (identical to RAGuard/ZKIP): if removing document i alone
     flips the full-set answer, document i is implicated.
  2. Pair pass -- SPECIFIC to genuine self-corroborating cliques, not "any
     flip-inducing pair": a first version implicated any (i, j) whose joint
     removal changed the answer, which turned out to over-trigger -- a pair
     consisting of one poison doc + one genuinely correct doc can *also*
     flip the vote when removed together, and excluding the correct doc
     alongside the poison one throws away real evidence (measured: this
     version's naive fix actually *raised* attack success, since it was
     just as likely to gut the correct side as the attacker's side). A
     colluding pair has a specific, checkable signature that an incidental
     pair doesn't: the two documents (a) share the *same* answer as each
     other, (b) that shared answer is *not* corroborated by any other
     retrieved document (exactly 2 documents hold it -- a self-contained
     clique, not a broader genuine consensus), and (c) that shared answer
     is what's currently winning (full_answer). Only a pair meeting all
     three is implicated; a single correct doc caught in an unrelated flip
     is never touched by this pass.
  3. Final answer = weighted_majority on the set with all implicated
     documents excluded (RAGuard/ZKIP's own recovery step, just over a
     larger exclusion set).

Coverage is probabilistic, not exhaustive: only MAX_PAIR_SAMPLES pairs are
sampled per query (bounding compute), so a colluding pair can be missed if
it's never sampled. This is an honest, documented limitation, not a claim of
perfect k'=2 detection.
"""
from __future__ import annotations
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .corpus import Document

MAX_PAIR_SAMPLES = 6


def weighted_majority(entries: List[Tuple[Document, float]]) -> Optional[str]:
    if not entries:
        return None
    tally = defaultdict(float)
    for doc, weight in entries:
        tally[doc.answer] += max(weight, 0.0) * doc.trust_score
    return max(tally.items(), key=lambda kv: kv[1])[0]


@dataclass
class GWCCResult:
    answer: Optional[str]
    calls: int
    flagged_subset: bool  # True iff any document was excluded


def gwcc_consensus(entries: List[Tuple[Document, float]], rng) -> GWCCResult:
    k = len(entries)
    calls = 0

    full_answer = weighted_majority(entries)
    calls += 1

    implicated: set = set()

    # Pass 1: singleton leave-one-out (RAGuard/ZKIP's own mechanism)
    for i in range(k):
        subset = entries[:i] + entries[i + 1:]
        a = weighted_majority(subset)
        calls += 1
        if a != full_answer:
            implicated.add(i)

    # Pass 2: pairwise leave-group-out, restricted to genuine self-corroborating
    # cliques (see module docstring for why "any flip-inducing pair" over-triggers).
    pairs = list(itertools.combinations(range(k), 2))
    rng.shuffle(pairs)
    answer_counts = defaultdict(int)
    for d, _ in entries:
        answer_counts[d.answer] += 1

    for (i, j) in pairs[:MAX_PAIR_SAMPLES]:
        calls += 1  # still charge for the counterfactual evaluation, even when
                    # the clique pre-check below skips computing it
        if i in implicated or j in implicated:
            continue
        di, dj = entries[i][0], entries[j][0]
        if di.answer != dj.answer:
            continue                          # not a corroborating pair at all
        if answer_counts[di.answer] != 2:
            continue                          # broader support than just this pair -> real consensus, not a clique
        if di.answer != full_answer:
            continue                          # this pair isn't even part of what's currently winning
        subset = [e for idx, e in enumerate(entries) if idx not in (i, j)]
        a = weighted_majority(subset)
        if a != full_answer:
            implicated.add(i)
            implicated.add(j)

    cleaned = [e for idx, e in enumerate(entries) if idx not in implicated]
    final = weighted_majority(cleaned) if cleaned else full_answer
    return GWCCResult(answer=final, calls=calls, flagged_subset=bool(implicated))
