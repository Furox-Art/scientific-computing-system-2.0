"""Statistical tests and summaries built on scipy.stats."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

__all__ = [
    "TestResult",
    "CorrelationResult",
    "DescribeResult",
    "describe",
    "t_test",
    "independent_t_test",
    "paired_t_test",
    "anova",
    "kruskal_wallis",
    "mann_whitney_u",
    "wilcoxon_signed_rank",
    "normality_test",
    "levene_test",
    "pearson_correlation",
    "spearman_correlation",
    "kendall_tau",
    "chi_square_independence",
    "cohens_d",
    "eta_squared_from_f",
    "cramers_v",
    "percentile",
    "z_scores",
    "norm_pdf",
    "norm_cdf",
    "norm_ppf",
    "BootstrapResult",
    "bootstrap_ci",
    "permutation_test",
]


@dataclass(frozen=True)
class TestResult:
    """Outcome of a statistical hypothesis test."""

    statistic: float
    p_value: float


@dataclass(frozen=True)
class CorrelationResult:
    """Correlation coefficient with its significance test."""

    r: float
    p_value: float


@dataclass(frozen=True)
class DescribeResult:
    """Descriptive summary of a numeric sample."""

    n: int
    mean: float
    std: float
    minimum: float
    q25: float
    median: float
    q75: float
    maximum: float
    skewness: float
    kurtosis: float


def _as_1d(x: object) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        msg = "expected a non-empty 1-D numeric sequence"
        raise ValueError(msg)
    return arr


def describe(data: object) -> DescribeResult:
    """Full descriptive summary (moments + quantiles) of a sample."""
    values = _as_1d(data)
    n = values.size
    mean = float(values.mean())
    centered = values - mean
    m2 = float(np.mean(centered * centered))
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    std = math.sqrt(m2 * n / (n - 1)) if n > 1 else 0.0
    skewness = m3 / m2**1.5 if m2 > 0.0 else 0.0
    kurtosis = m4 / (m2 * m2) - 3.0 if m2 > 0.0 else -3.0
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return DescribeResult(
        n=int(n),
        mean=mean,
        std=std,
        minimum=float(values.min()),
        q25=float(q25),
        median=float(median),
        q75=float(q75),
        maximum=float(values.max()),
        skewness=skewness,
        kurtosis=kurtosis,
    )


def t_test(sample: object, popmean: float = 0.0) -> TestResult:
    """One-sample t-test of the sample mean against ``popmean``."""
    result = sps.ttest_1samp(_as_1d(sample), popmean=popmean)
    return TestResult(float(result.statistic), float(result.pvalue))


def independent_t_test(a: object, b: object, equal_var: bool = True) -> TestResult:
    """Two-sample t-test; ``equal_var=False`` runs Welch's variant."""
    res = sps.ttest_ind(_as_1d(a), _as_1d(b), equal_var=equal_var)
    return TestResult(float(res.statistic), float(res.pvalue))


def paired_t_test(a: object, b: object) -> TestResult:
    """Paired-samples t-test on two related samples."""
    res = sps.ttest_rel(_as_1d(a), _as_1d(b))
    return TestResult(float(res.statistic), float(res.pvalue))


def anova(*groups: object) -> TestResult:
    """One-way analysis of variance F-test across two or more groups."""
    if len(groups) < 2:
        msg = "anova needs at least two groups"
        raise ValueError(msg)
    prepared = [_as_1d(g) for g in groups]
    res = sps.f_oneway(*prepared)
    return TestResult(float(res.statistic), float(res.pvalue))


def kruskal_wallis(*groups: object) -> TestResult:
    """Non-parametric Kruskal-Wallis H-test across groups."""
    if len(groups) < 2:
        msg = "kruskal_wallis needs at least two groups"
        raise ValueError(msg)
    res = sps.kruskal(*[_as_1d(g) for g in groups])
    return TestResult(float(res.statistic), float(res.pvalue))


def mann_whitney_u(a: object, b: object) -> TestResult:
    """Mann-Whitney U rank test comparing two independent samples."""
    res = sps.mannwhitneyu(_as_1d(a), _as_1d(b), alternative="two-sided")
    return TestResult(float(res.statistic), float(res.pvalue))


def wilcoxon_signed_rank(a: object, b: object) -> TestResult:
    """Wilcoxon signed-rank test for paired samples."""
    res = sps.wilcoxon(_as_1d(a), _as_1d(b))
    return TestResult(float(res.statistic), float(res.pvalue))


def normality_test(data: object) -> TestResult:
    """Shapiro-Wilk test of normality."""
    res = sps.shapiro(_as_1d(data))
    return TestResult(float(res.statistic), float(res.pvalue))


def levene_test(*groups: object) -> TestResult:
    """Levene test for equality of variances across groups."""
    if len(groups) < 2:
        msg = "levene_test needs at least two groups"
        raise ValueError(msg)
    res = sps.levene(*[_as_1d(g) for g in groups])
    return TestResult(float(res.statistic), float(res.pvalue))


def pearson_correlation(x: object, y: object) -> CorrelationResult:
    """Pearson linear correlation between two samples."""
    res = sps.pearsonr(_as_1d(x), _as_1d(y))
    return CorrelationResult(r=float(res.statistic), p_value=float(res.pvalue))


def spearman_correlation(x: object, y: object) -> CorrelationResult:
    """Spearman rank correlation between two samples."""
    res = sps.spearmanr(_as_1d(x), _as_1d(y))
    return CorrelationResult(r=float(res.statistic), p_value=float(res.pvalue))


def kendall_tau(x: object, y: object) -> CorrelationResult:
    """Kendall's tau rank correlation between two samples."""
    res = sps.kendalltau(_as_1d(x), _as_1d(y))
    return CorrelationResult(r=float(res.statistic), p_value=float(res.pvalue))


def chi_square_independence(table: object) -> TestResult:
    """Chi-square test of independence on a contingency table."""
    contingency = np.asarray(table, dtype=float)
    if contingency.ndim != 2:
        msg = "table must be a 2-D contingency matrix"
        raise ValueError(msg)
    stat, p_value, _dof, _expected = sps.chi2_contingency(contingency)
    return TestResult(float(stat), float(p_value))


def cohens_d(a: object, b: object) -> float:
    """Cohen's d standardized mean difference with pooled SD."""
    x, y = _as_1d(a), _as_1d(b)
    nx, ny = x.size, y.size
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / pooled)


def eta_squared_from_f(f_statistic: float, df1: int, df2: int) -> float:
    """Eta-squared effect size derived from a one-way ANOVA F statistic."""
    return float((f_statistic * df1) / (f_statistic * df1 + df2))


def cramers_v(table: object) -> float:
    """Cramer's V association strength from a contingency table in [0, 1]."""
    contingency = np.asarray(table, dtype=float)
    observed = np.atleast_2d(contingency).astype(float)
    stat, _p, _dof, _expected = sps.chi2_contingency(observed)
    n = observed.sum()
    phi2 = stat / n
    rows, cols = observed.shape
    denominator = min(rows - 1, cols - 1)
    if denominator <= 0:
        msg = "contingency table needs at least two rows and two columns"
        raise ValueError(msg)
    return float(np.sqrt(phi2 / denominator))


def percentile(data: object, q: float | list[float]) -> float | list[float]:
    """Percentile(s) of a sample for ``q`` given in [0, 100]."""
    values = _as_1d(data)
    if isinstance(q, list):
        return [float(v) for v in np.percentile(values, q)]
    return float(np.percentile(values, q))


def z_scores(data: object) -> np.ndarray:
    """Standardize a sample to zero mean and unit standard deviation (ddof=1)."""
    values = _as_1d(data)
    sd = np.std(values, ddof=1)
    if sd == 0:
        msg = "z_scores undefined for a constant sample"
        raise ValueError(msg)
    return np.asarray((values - np.mean(values)) / sd)


def norm_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal probability density function."""
    return float(sps.norm.pdf(x, loc=mu, scale=sigma))


def norm_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal cumulative distribution function."""
    return float(sps.norm.cdf(x, loc=mu, scale=sigma))


def norm_ppf(q: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Inverse normal CDF (quantile function)."""
    return float(sps.norm.ppf(q, loc=mu, scale=sigma))


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap estimate with standard error and percentile interval."""

    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float


def bootstrap_ci(
    data: object,
    statistic: Callable[..., float] | None = None,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Percentile bootstrap confidence interval for an arbitrary statistic."""
    values = _as_1d(data)
    if not 0.0 < confidence < 1.0:
        msg = "confidence must be in (0, 1)"
        raise ValueError(msg)
    stat_fn = statistic if statistic is not None else np.mean
    n = values.size
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n, size=(n_resamples, n))
    samples = values[resample_indices]
    try:
        estimates = np.asarray(stat_fn(samples, axis=1), dtype=float)
    except TypeError:
        estimates = np.array([float(stat_fn(sample)) for sample in samples])
    point_estimate = float(stat_fn(values))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return BootstrapResult(
        estimate=point_estimate,
        standard_error=float(np.std(estimates, ddof=1)),
        ci_low=float(low),
        ci_high=float(high),
    )


def permutation_test(
    a: object,
    b: object,
    n_permutations: int = 10_000,
    seed: int | None = None,
) -> TestResult:
    """Two-sided permutation test on the difference of means."""
    group_a = _as_1d(a)
    group_b = _as_1d(b)
    pooled = np.concatenate([group_a, group_b])
    nx = group_a.size
    observed = float(group_a.mean() - group_b.mean())
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((n_permutations, pooled.size)), axis=1)
    shuffled = pooled[order]
    permuted_diffs = shuffled[:, :nx].mean(axis=1) - shuffled[:, nx:].mean(axis=1)
    extreme_count = int(np.count_nonzero(np.abs(permuted_diffs) >= abs(observed)))
    p_value = (extreme_count + 1) / (n_permutations + 1)
    return TestResult(statistic=observed, p_value=float(p_value))
