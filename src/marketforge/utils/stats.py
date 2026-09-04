"""Shared salary statistics helpers.

Extracted so every place that computes a salary percentile from raw job
rows — SalaryIntelligenceAgent, MarketAnalystLeadAgent's per-role inline
block, and api/main.py's ad-hoc filtered /market/salary endpoint — applies
the same two-layer outlier defense instead of three drifting copies of it.
"""
from __future__ import annotations

# Minimum sample size before a percentile is reported at all. Below this,
# a single outlier or a handful of rows can swing a "percentile" wildly;
# matches the n>=10 discipline already used for sponsorship segments.
MIN_SALARY_SAMPLE_SIZE = 10

# Layer 1: fixed absolute-bound filter on the salary midpoint. Catches
# wildly implausible values (e.g. a mis-parsed day-rate) regardless of
# sample shape, before IQR trimming even runs.
SALARY_MIDPOINT_MIN = 20_000
SALARY_MIDPOINT_MAX = 300_000


def iqr_trim(values: list[float]) -> list[float]:
    """Reject points outside Q1-1.5*IQR / Q3+1.5*IQR, computed from `values` itself."""
    s = sorted(values)
    n = len(s)
    q1 = s[max(0, int(n * 0.25) - 1)]
    q3 = s[max(0, int(n * 0.75) - 1)]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [v for v in s if lo <= v <= hi]


def clean_salary_midpoints(raw_midpoints: list[float]) -> list[float]:
    """Apply both outlier-defense layers to a list of raw salary midpoints."""
    bounded = [m for m in raw_midpoints if SALARY_MIDPOINT_MIN <= m <= SALARY_MIDPOINT_MAX]
    return iqr_trim(bounded) if len(bounded) >= MIN_SALARY_SAMPLE_SIZE else bounded


def percentile(sorted_values: list[float], p: float) -> float | None:
    """p in [0, 100]. Returns None if sample is below MIN_SALARY_SAMPLE_SIZE."""
    n = len(sorted_values)
    if n < MIN_SALARY_SAMPLE_SIZE:
        return None
    idx = max(0, int(n * p / 100) - 1)
    return round(sorted_values[idx])
