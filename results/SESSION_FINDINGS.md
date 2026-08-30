# Session findings — continuing from the handed-off Path A codebase

## What was verified as-is

The uploaded `omniguard_rag_fixed.zip` was extracted and run before any changes were
made. Its single-seed benchmark output matched `walkthrough.md`'s own documented
table exactly, confirming the handoff was genuine, working code, not a description of
code. `ablations.py` and `stats_utils.py` — flagged as missing earlier in this
project's history — are present in this zip and both import and run correctly.

## A real bug found in the project's own central contribution

`unified_rag_defense/gwcc_consensus.py` (Ring 3, the project's headline novel
mechanism) computed its final answer by taking a **majority vote across
counterfactual subsets** — the full retrieved set, every single-document removal, and
several sampled pair removals, each counted as one equal-weight "ballot."

Measured directly (`run_gwcc_diagnostic.py`, k_poison swept from 3 to 12): **this
vote never once differed from a plain, single-pass vote on the same retrieved set —
0 divergences out of 653 tested cases.** The mechanism was real code, genuinely
running, genuinely costing extra calls — and having zero effect on the answer.

**Why:** when k'≥2 colluding documents already hold a plurality in the full top-k
set, only the small number of counterfactual subsets that happen to remove *enough*
of them flip away from the poison answer. A worked example from this corpus: 3
colluding poison docs + 2 clean docs in a top-5 retrieval — only 4 of 15 sampled
removals flipped the vote. Since the aggregate vote just counts ballots, the
poison answer wins the "majority-of-majorities" too, for the same reason it won
the original vote.

## The fix

Rewrote `gwcc_consensus.py` to do actual leave-group-out **exclusion** (the real
logic RAGuard/ZKIP's own single-document LOO uses — "does removing this change the
answer? If so, distrust it and recompute" — genuinely generalized to pairs):

1. Singleton pass: identical to RAGuard/ZKIP.
2. Pair pass: **restricted to genuine self-corroborating cliques** — both documents
   must share the same answer, that answer must have *no other* support among the
   retrieved set (a self-contained pair, not a broader real consensus), and it must
   be what's currently winning. An earlier, looser version of this fix (flag any
   pair whose removal changes the vote) was tried first and rejected: it *raised*
   attack success, because it excluded correct-but-numerically-scarce evidence just
   as often as it excluded actual poison pairs.
3. Final answer = plain vote on the set with all implicated documents removed.

Re-measured after the fix: genuine divergence from plain voting (11–92 divergences
out of 123–191 escalations, depending on k_poison) — Ring 3 is no longer a no-op.

## The honest limitation this surfaces

Even fixed, single-query GWCC has a real ceiling. Tested in isolation (trust store
reset every call, i.e. Ring 0–3 with no memory across queries), stealth-collusion
attack success is **8.5%**, not 0% — because a self-contained pair of documents
agreeing on the *correct* answer and a self-contained pair of documents colluding
on a *false* answer are, within a single query, statistically identical in
structure. No purely-structural, single-query heuristic can perfectly tell them
apart; that's not a coding bug, it's a real information-theoretic limit of the
approach as specified.

**What actually produces this system's 0% stealth-ASR headline number:** the
cross-query Dynamic Trust Store. Same fixed Ring 0–3, same attacks, only the trust
store toggled:

| Configuration | Accuracy | Overall ASR | Stealth ASR |
|---|---|---|---|
| Ring 0–3 (fixed GWCC), trust store reset every call | 98.8% | 1.4% | 8.5% |
| Ring 0–3 (fixed GWCC), trust store persists across queries | 100.0% | 0.0% | 0.0% |

The trust store repeatedly downweights documents that keep behaving suspiciously
across many queries — camouflage that survives one query's structural analysis
doesn't survive being seen (and voted against) dozens of times. That's a legitimate
mechanism, but the report should credit it explicitly rather than attributing the
result to GWCC alone.

## Final 8-seed evaluation (post-fix)

`results/path_a_report.md`, regenerated after the fix, 8 seeds × 200 queries:

- OmniGuard-RAG (Ours): **100.0±0.0%** accuracy, **0.0±0.0%** overall ASR,
  stealth ASR **0.1±0.1%** (small, honest, nonzero — not a fabricated perfect zero).
- Ablation ladder's last row, `+Ring2 (both signals)` (Ring 0–3, fixed GWCC,
  deliberately excluding the trust store by the ladder's own design): **98.6±0.2%**
  accuracy, stealth ASR **9.8±1.3%** — consistent with the isolated measurement
  above, and now an *informative* ablation step instead of a masked no-op.

## Files touched this session

- `unified_rag_defense/gwcc_consensus.py` — bug fixed (see above); everything else
  in the codebase was verified, not modified.
- `walkthrough.md` — added a correction note pointing here, rather than silently
  rewriting its original claim.
- `results/` — regenerated fresh (`path_a_report.md`, `path_a_raw_results.json`,
  `gwcc_diagnostic.md`) against the fixed code.
