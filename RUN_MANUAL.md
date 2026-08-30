# OmniGuard-RAG — Manual: Running the Complete Project on Your Desktop

*Every command below was actually run against the real extracted project
during this session (not assumed) — including exact timings, so the "how
long will this take" numbers are measured, not guessed. Your machine will
differ somewhat, but not by an order of magnitude.*

---

## 0. What you need

- **Python 3.9 or newer** (tested and confirmed working on 3.12; nothing in
  the code requires a newer version, but older than 3.9 isn't tested).
- **Three third-party packages**: `numpy`, `scikit-learn`, `scipy`. Nothing
  else — no LLM API key, no internet connection needed once installed, no
  GPU. There is no `requirements.txt` shipped with the project, so you'll
  type the install command yourself (Step 2 below).
- No spiral binding, no LaTeX, none of your report-formatting tools are
  needed for any of this — this manual is purely about running the code.

**Check your Python version first:**

Open a terminal (Command Prompt / PowerShell on Windows, Terminal on
macOS/Linux) and run:

```bash
python3 --version
```

On Windows, if `python3` isn't recognized, try `python --version` instead —
Windows installs often register the command as just `python`. Use whichever
one responds for every command below (substitute `python` for `python3`
throughout if that's what worked here).

---

## 1. Locate your extracted folder

You said you've already unzipped it. Find the folder that directly
contains `run_omniguard_benchmark.py` and a subfolder called
`unified_rag_defense/` — that inner folder is your project root. It's
usually named `OmniGuard-RAG/`. **Every command in this manual assumes your
terminal is inside that folder.**

Navigate there:

```bash
cd path/to/OmniGuard-RAG
```

Confirm you're in the right place — this should list the four run scripts
and the `unified_rag_defense` folder:

```bash
# macOS/Linux:
ls
# Windows PowerShell:
dir
```

You should see:
```
run_omniguard_benchmark.py
run_gwcc_diagnostic.py
run_full_evaluation.py
run_embedding_comparison.py
unified_rag_defense/
results/
walkthrough.md
```

If any of those are missing, you're either in the wrong folder or your zip
extracted into a nested folder (e.g. `OmniGuard-RAG/OmniGuard-RAG/`) — check
one level down.

---

## 2. Set up a virtual environment and install dependencies

**Do this even if you already have numpy/scikit-learn/scipy installed
globally** — a virtual environment keeps this project isolated and sidesteps
a common error (Step 6, error #1) that some systems throw otherwise. This
only needs to be done once.

```bash
# macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# Windows PowerShell:
python -m venv venv
venv\Scripts\Activate.ps1

# Windows Command Prompt:
python -m venv venv
venv\Scripts\activate.bat
```

Your terminal prompt should now show `(venv)` at the start of the line —
that confirms it's active. Now install the three packages:

```bash
pip install numpy scikit-learn scipy
```

This takes under a minute on a normal connection. You'll need to reactivate
this environment (`source venv/bin/activate` or the Windows equivalent)
every time you open a new terminal to work on this project — you won't need
to reinstall anything, just reactivate.

---

## 3. Verify your setup with the fastest script first

Before running anything long, confirm everything is wired correctly with
the single-seed benchmark. **Measured runtime: ~35 seconds.**

```bash
python3 run_omniguard_benchmark.py
```

You should see a results table print directly to your terminal, ending
with a line for `OmniGuard-RAG (Ours)`. If this runs cleanly, your setup is
correct and every other script will work too — skip ahead to whichever
script you actually need. If it errors, go straight to Step 6.

---

## 4. Running each script

Run these in any order — they're independent — but this is a sensible
progression from fastest to slowest.

### 4a. Main single-seed benchmark (already run in Step 3)

```bash
python3 run_omniguard_benchmark.py
```
**What it does:** Runs the 6-system comparison (Vanilla RAG, DRS-only,
ShieldRAG-only, RAGuard/ZKIP, TriShield, OmniGuard-RAG) against all five
attack types, once, with a fixed seed. Prints the table to your terminal —
**does not write any file.**
**Measured time: ~35 seconds.**

### 4b. GWCC diagnostic

```bash
python3 run_gwcc_diagnostic.py
```
**What it does:** Sweeps the number of colluding poison documents
(k_poison = 3, 5, 8, 12) and directly measures whether Ring 3's consensus
mechanism changes the answer versus plain voting on the same retrieved set.
This is the script that verifies the GWCC bug-fix actually works (see
Section 4.2 of your explanatory README).
**Writes:** `results/gwcc_diagnostic.md`
**Measured time: ~5 seconds.**

### 4c. Full multi-seed evaluation (the numbers your report should cite)

This is the one that produces the 8-seed, confidence-interval numbers used
throughout your report's Results section.

**Recommended first pass — quick mode (3 seeds, fewer queries):**
```bash
python3 run_full_evaluation.py --quick
```
**Measured time: ~47 seconds.**

**Full default run (8 seeds × 200 queries — the numbers actually cited in
your README):**
```bash
python3 run_full_evaluation.py
```
**Measured time: ~7 minutes** (one seed at full scale took 52 seconds in
testing; the default runs 8 seeds).

**Important — `--quick` and the full run are not combinable.** Quick mode
uses 60 queries/seed; the default uses 200. The script checkpoints by seed,
but only resumes a run using the *same* `n_queries` and `docs_per_topic`
settings — running `--quick` first and the full command afterward will
**restart from scratch**, not extend the quick results. If you're short on
time, quick mode alone is enough to sanity-check the system; use the full
default run when you actually need report-quality numbers.

**If it gets interrupted partway through** (closed terminal, laptop slept,
etc.), just run the exact same command again — it checkpoints per seed and
picks up where it left off. You'll see a message like:
```
5 of the default 8 seeds not yet run: [41, 59, 79, 97, 113]
```
That's normal, not an error — it's telling you it's resuming.

**Writes:** `results/path_a_raw_results.json`, `results/path_a_report.md`
(this second file is the one with the exact tables quoted in your
explanatory README's Section 5).

**Optional flags**, if you want to experiment beyond the defaults:
```bash
python3 run_full_evaluation.py --seeds 7 11 23      # only these 3 seeds
python3 run_full_evaluation.py --n-queries 100      # override queries/seed
python3 run_full_evaluation.py --docs-per-topic 20  # smaller/larger corpus
```

### 4d. Embedding comparison (Path B — TF-IDF vs. LSA)

This is the script flagged as "not yet run" in your earlier session — the
one your explanatory README lists as an open thread / future-work item.

```bash
python3 run_embedding_comparison.py
```
**What it does:** Re-runs the same ring-ablation ladder as 4c, but twice —
once with TF-IDF embeddings, once with LSA (`TruncatedSVD`) embeddings —
and reports both side by side, to test whether Ring 1's detection advantage
(discussed in your README's Section 5 limitations) is specific to TF-IDF's
small closed vocabulary or holds in a richer embedding space.
**Default: 3 seeds × 200 queries.** Estimated from a measured partial run
(1 seed × 40 queries took 21 seconds): **roughly 5–6 minutes** for the full
default. Same checkpoint/resume behavior as 4c.

**Writes:** `results/path_b_raw_results.json`, `results/path_b_report.md`

**Optional flag:**
```bash
python3 run_embedding_comparison.py --recalibrate-risk-threshold
```
Recalibrates the risk router's thresholds separately for each embedding
space instead of reusing TF-IDF's calibrated values for both — use this if
you want to check whether LSA's numbers change when it gets its own
properly-calibrated threshold rather than TF-IDF's.

---

## 5. Where everything lands

All output stays inside your project folder's `results/` subfolder,
**regardless of which directory your terminal was in when you ran the
command** — every script resolves this path relative to its own location,
not your current directory, so you'll never lose track of an output file
because you ran a command from the wrong place.

| File | Written by |
|---|---|
| `results/gwcc_diagnostic.md` | `run_gwcc_diagnostic.py` |
| `results/path_a_report.md`, `results/path_a_raw_results.json` | `run_full_evaluation.py` |
| `results/path_b_report.md`, `results/path_b_raw_results.json` | `run_embedding_comparison.py` |
| `results/SESSION_FINDINGS.md`, `results/PATH_A_SUMMARY.md` | already shipped in your zip — written during earlier development, not regenerated by these scripts |

`run_omniguard_benchmark.py` (4a) only prints to the terminal — it writes
no file.

**Note:** re-running 4c or 4d overwrites their existing `results/` files
with fresh numbers. If your zip's shipped `path_a_report.md` already has
numbers you're planning to cite and you just want to *verify* them rather
than regenerate them, you don't need to run 4c at all — just open the file
that's already there. Re-run it only if you want to confirm reproducibility
yourself, or if you've changed a parameter (like `--docs-per-topic`) and
want fresh numbers for that configuration.

---

## 6. Troubleshooting

**Error: `error: externally-managed-environment` when running `pip
install`**
This happens on some Linux distributions (Debian/Ubuntu and similar) that
block installing packages outside a virtual environment. This is exactly
what Step 2's virtual environment avoids — if you skipped it, go back and
create one now. If you specifically don't want to use a venv, the
workaround is `pip install numpy scikit-learn scipy --break-system-packages`,
but the venv approach is safer and is what this manual recommends.

**Error: `ModuleNotFoundError: No module named 'numpy'` (or `sklearn`, or
`scipy`)**
Your virtual environment isn't active, or the install in Step 2 didn't
complete. Check your terminal prompt for `(venv)` at the start of the
line — if it's not there, run the activation command from Step 2 again,
then re-run `pip install numpy scikit-learn scipy`.

**Error: `ModuleNotFoundError: No module named 'unified_rag_defense'`**
You're not in the project root, or the `unified_rag_defense/` folder didn't
extract alongside the run scripts. Re-check Step 1 — `ls`/`dir` should show
`unified_rag_defense/` as a sibling of `run_omniguard_benchmark.py`, not
nested inside another folder.

**Windows PowerShell: `venv\Scripts\Activate.ps1 cannot be loaded because
running scripts is disabled on this system`**
Run this once in an **administrator** PowerShell window, then retry
activation:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**A script prints results but "hangs" / seems stuck for a while**
The multi-seed scripts (4c, 4d) genuinely take minutes — see the measured
timings above. If you're past double the estimated time with no output
change at all, something's wrong; otherwise, it's just working.

**Numbers differ slightly from the ones in your explanatory README**
Small differences (a fraction of a percentage point) between separate runs
of the *same* seeds are not expected — every random process in this
codebase is explicitly seeded (see the README's Section 2.2 on
reproducibility, and the note there about the earlier `hash()`-based bug
that used to cause exactly this). If you get identical seeds but different
numbers, make sure you're not accidentally running with a different
`--docs-per-topic` or `--n-queries` value than the default.

---

## 7. Quick reference — copy/paste block

Once your venv is set up (Steps 1–2, one-time only):

```bash
cd path/to/OmniGuard-RAG
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1

python3 run_omniguard_benchmark.py      # ~35s  — sanity check, prints only
python3 run_gwcc_diagnostic.py          # ~5s   — writes gwcc_diagnostic.md
python3 run_full_evaluation.py          # ~7min — writes path_a_report.md (your report's main numbers)
python3 run_embedding_comparison.py     # ~5-6min — writes path_b_report.md (open thread from your last session)
```
