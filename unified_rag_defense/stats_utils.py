"""
stats_utils.py — Student's-t 95% confidence intervals across independent
seeds (Path A).

WHY THIS EXISTS: run_full_evaluation.py runs every system/variant at
several independent seeds (each an independently regenerated corpus, query
set, and DRS calibration split -- not a reshuffle of the same data) and
needs to report each metric as mean ± 95% CI, not just a bare mean or "here
are N numbers, they look similar." A confidence interval computed with a
fixed z=1.96 (the normal-distribution shortcut) understates the interval at
small seed counts -- exactly the regime this project runs in (8 seeds by
default, 3 in --quick mode). Student's-t with n-1 degrees of freedom is the
textbook-correct choice for a small-sample mean with unknown population
variance, and is what run_full_evaluation.py's own report text claims
("Student's-t") -- so that claim needs to actually be true of the number
printed next to it, not merely a familiar-sounding label attached to a
z-interval.

EDGE CASE -- n=1: a single-seed call (e.g. someone running with
--seeds 7 alone, or a future single-seed sanity script reusing this module)
has zero degrees of freedom for a t-interval -- variance and a CI are
undefined, not just noisy. Rather than crash or silently print a fabricated
interval, summarize() reports ci=0.0 for n=1 (a bare point estimate), which
is honest about what a single observation can and cannot support. This
mirrors how holdout_fpr is reported per-seed then summarized the same way
as every other metric, including at n=1 in a single-seed diagnostic run.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from scipy import stats as _scipy_stats


@dataclass
class Stat:
    mean: float
    ci: float  # half-width of the 95% CI; 0.0 when n<2 (undefined, not fabricated)
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.1f}±{self.ci:.1f}"


def summarize(values: List[float]) -> Stat:
    """Mean and 95% Student's-t CI half-width across `values` (one number
    per seed). n=0 returns a zero Stat (nothing was measured); n=1 returns
    the bare value with ci=0.0 (see module docstring); n>=2 computes a real
    t-interval using the sample standard deviation (ddof=1, the unbiased
    estimator -- population variance is unknown here, only estimated from
    the seeds actually run) and the two-tailed 97.5th-percentile critical
    value of Student's-t with n-1 degrees of freedom."""
    n = len(values)
    if n == 0:
        return Stat(mean=0.0, ci=0.0, n=0)
    mean = sum(values) / n
    if n == 1:
        return Stat(mean=mean, ci=0.0, n=1)
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = variance ** 0.5
    t_crit = float(_scipy_stats.t.ppf(0.975, df=n - 1))
    ci = t_crit * std / (n ** 0.5)
    return Stat(mean=mean, ci=ci, n=n)
