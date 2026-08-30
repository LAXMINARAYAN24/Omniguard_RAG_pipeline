# Path A — Summary

Path A's goal was to make OmniGuard-RAG's results defensible: per-ring
ablations, confidence intervals across independent seeds, and real
wall-clock latency instead of call-counts alone. This file ties the three
output documents together and states plainly where a finding along the way
complicates a claim already written into `walkthrough.md`.

## What's here

- **`path_a_report.md`** — the main deliverable. Mean ± 95% CI (Student's-t,
  `unified_rag_defense/stats_utils.py`) across 8 independent seeds
  (`[7, 11, 23, 41, 59, 79, 97, 113]`, 200 queries/seed, 1,600 queries total
  per system). Table 1 is the 6-system comparison; Table 2 is the per-ring
  ablation ladder; Table 3 is compute cost, calls vs. measured latency.
- **`path_a_raw_results.json`** — every per-seed value behind Table 1/2's
  means and CIs, plus the per-seed holdout-FPR values, so the aggregation
  in `path_a_report.md` can be checked or recomputed independently rather
  than taken on faith.
- **`gwcc_diagnostic.md`** — a targeted follow-up (below).

## The per-ring ablation ladder (Table 2)

Each row in Table 2 adds exactly one ring to the previous row's pipeline
(`unified_rag_defense/ablations.py`'s module docstring has the full
per-step breakdown). Two things worth calling out directly:

**Ring 0 alone already reaches 99.4% accuracy**, nearly matching the full
system's 100%. This is a real, verified effect, not a ladder artifact: it's
driven entirely by the PIDP regime, where accuracy jumps from Vanilla RAG's
0% (a complete wipeout — undefended query-path poisoning always wins) to
100% once Ring 0's suffix stripper runs, while every other regime is
byte-identical between "Vanilla RAG" and "Ring0 alone." One regime flipping
100 percentage points, out of seven regimes, accounts for the entire
~14-point gap. Verified directly by comparing per-regime accuracy between
the two systems at seed=7, not inferred from the aggregate.

**The ladder deliberately excludes the DynamicTrustStore.** The last row,
"+Ring2 (both signals)," uses the exact same Ring0→Ring1→Ring2→Ring3 logic
`omniguard_pipeline.run_omniguard` uses — but without the persistent
cross-query trust store the full pipeline also has. That's intentional
(mixing a per-ring effect with a cross-query effect wouldn't be a clean
ladder), and the gap between this row and "OmniGuard-RAG (Ours)" in Table 1
is the trust store's own contribution. At 8 seeds: accuracy gap is small
(99.8% vs. 100.0%), but avg_calls drops more noticeably (2.1 vs. 1.3) —
the trust store's main measured effect is that it reduces how often later
queries need to escalate to Ring 3 at all, not primarily accuracy.

## GWCC diagnostic — a finding that narrows a claim in `walkthrough.md`

`walkthrough.md` §3.5 states: *"GWCC catches something the other rings
can't ... it resolves cases the fast path alone would not."* That claim has
two separable parts:

1. Ring 3 gets invoked specifically because of the contention signal, on
   cases cohesion alone would wave through. **Verified true**: with only
   the cohesion signal active, `collusion_stealth` never escalates to Ring
   3 (0/200 at seed=7); with both signals active, it escalates in exactly
   the queries where poison reaches the top-5 retrieval window (119/200).
2. GWCC's own consensus mechanism, once invoked, recovers a different
   (correct) answer than plain weighted-majority voting on that identical
   retrieved set would have. **Tested directly, found false**: across every
   escalation in the full 8-seed benchmark (main comparison + ablation
   ladder combined) and, more thoroughly, across a deliberately
   strengthened `collusion_stealth` attack up to `k_poison=12`
   (`run_gwcc_diagnostic.py`, `results/gwcc_diagnostic.md`), GWCC's verdict
   **never once** diverges from plain voting on the same input — 0/654
   escalations tested at seed=7 alone across four attack-strength levels,
   despite the top-5 window frequently containing a poison plurality or
   majority by the higher `k_poison` levels.

**Why, most likely** (see `run_gwcc_diagnostic.py`'s docstring for the full
argument): GWCC's final answer is a plurality vote across 12
sub-evaluations (1 full-set + 5 leave-one-out + 6 sampled leave-pair-out at
k=5), each itself a weighted-majority vote. If poison holds enough combined
weight to win the full-set vote, removing just 1–2 of the 5 retrieved
documents at a time often still leaves poison holding a plurality within
that sub-evaluation too — so poison tends to win most of the 12 sub-votes,
not just the full one. This is a property of the aggregation rule
(plurality-of-sub-majorities) itself, not of TF-IDF vs. real embeddings.

**What this does and does not mean:**
- It does **not** mean Ring 2's contention signal is wrong — that detection
  is independently verified above (part 1).
- It does **not** mean OmniGuard-RAG's measured 0% stealth-collusion ASR is
  fake — it's real and reproducible across all 8 seeds (Table 1).
- It **does** mean that 0% is currently earned by (a) DRS/Ring 1 catching
  non-stealth collusion before Ring 2/3 ever engage, and (b) the
  DynamicTrustStore reshaping which documents reach the top-5 window on
  later queries — **not** by GWCC's consensus step recovering an answer the
  fast path would have missed on the same input, on the attack strength
  this benchmark currently tests.

This is a narrower, more precisely-supported claim than the one currently
written in `walkthrough.md` §3.5, and closing that gap — either in the
write-up's wording (say what part 1 and part 2 actually established,
separately) or in GWCC's own aggregation rule (an aggregation that weighs
sub-vote *agreement* rather than a flat plurality-of-pluralities might
behave differently at high poison density; untested here) — is worth doing
before this goes in the final report.

## A bug found and fixed along the way

`run_full_evaluation.py`'s checkpoint-resume logic had a pre-existing bug:
resuming with new seeds after a prior partial run duplicated every newly-run
seed number in the JSON checkpoint's `seeds` list and the printed seed list
(e.g. resuming with `[41, 59]` after `[7, 11, 23]` produced
`seeds=[7, 11, 23, 41, 59, 41, 59]`). The per-seed *value* lists that
Table 1/2's means and CIs are actually computed from were unaffected (each
value is appended exactly once per seed, inside the same loop, independent
of the buggy line) — so no number in `path_a_report.md` was ever wrong —
but the seeds list itself, which is what a reader checks to know which
seeds back a given report, was. Fixed in `run_full_evaluation.py` (see its
inline comment at the fix site); the already-written JSON checkpoint was
repaired and re-verified against its value-list lengths before the final
8-seed run completed.
