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
    "StreamingStats",
    "bootstrap_ci",
    "permutation_test",
    "covariance_matrix",
    "correlation_matrix",
    "multivariate_normal_logpdf",
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


def _as_1d(x: object, *, min_size: int = 1) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size < min_size:
        raise ValueError(
            "expected a non-empty 1-D numeric sequence"
            if min_size == 1
            else f"expected a 1-D numeric sequence with at least {min_size} value(s)"
        )
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("sample values must be finite")
    return arr


def _paired_arrays(a: object, b: object, *, min_size: int = 2) -> tuple[np.ndarray, np.ndarray]:
    x = _as_1d(a)
    y = _as_1d(b)
    if x.shape != y.shape:
        raise ValueError("paired samples must have the same length")
    if x.size < min_size:
        raise ValueError(f"paired samples need at least {min_size} observations")
    return x, y


def _validate_normal(mu: float, sigma: float) -> None:
    if not np.isfinite(mu):
        raise ValueError("mu must be finite")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive and finite")


def _contingency_table(table: object) -> np.ndarray:
    observed = np.asarray(table, dtype=float)
    if observed.ndim != 2:
        raise ValueError("table must be a 2-D contingency matrix")
    if min(observed.shape) < 2:
        raise ValueError("contingency table needs at least two rows and two columns")
    if not bool(np.all(np.isfinite(observed))) or np.any(observed < 0.0):
        raise ValueError("contingency counts must be finite and non-negative")
    if float(np.sum(observed)) <= 0.0:
        raise ValueError("contingency table must contain a positive total count")
    return observed


def _matrix_observations(data: object) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("data must be a 2-D matrix with at least two observations")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("data must contain only finite values")
    return values


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
    if not np.isfinite(popmean):
        raise ValueError("popmean must be finite")
    result = sps.ttest_1samp(_as_1d(sample, min_size=2), popmean=popmean)
    return TestResult(float(result.statistic), float(result.pvalue))


def independent_t_test(a: object, b: object, equal_var: bool = True) -> TestResult:
    """Two-sample t-test; ``equal_var=False`` runs Welch's variant."""
    res = sps.ttest_ind(_as_1d(a, min_size=2), _as_1d(b, min_size=2), equal_var=equal_var)
    return TestResult(float(res.statistic), float(res.pvalue))


def paired_t_test(a: object, b: object) -> TestResult:
    """Paired-samples t-test on two related samples."""
    x, y = _paired_arrays(a, b)
    res = sps.ttest_rel(x, y)
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
    x, y = _paired_arrays(a, b)
    res = sps.wilcoxon(x, y)
    return TestResult(float(res.statistic), float(res.pvalue))


def normality_test(data: object) -> TestResult:
    """Shapiro-Wilk test of normality; requires at least three observations."""
    res = sps.shapiro(_as_1d(data, min_size=3))
    return TestResult(float(res.statistic), float(res.pvalue))


def levene_test(*groups: object) -> TestResult:
    """Levene test for equality of variances across groups."""
    if len(groups) < 2:
        msg = "levene_test needs at least two groups"
        raise ValueError(msg)
    res = sps.levene(*[_as_1d(g) for g in groups])
    return TestResult(float(res.statistic), float(res.pvalue))


def pearson_correlation(x: object, y: object) -> CorrelationResult:
    """Correlation between equal-length finite samples."""
    a, b = _paired_arrays(x, y)
    res = sps.pearsonr(a, b)
    return CorrelationResult(r=float(res.statistic), p_value=float(res.pvalue))


def spearman_correlation(x: object, y: object) -> CorrelationResult:
    """Correlation between equal-length finite samples."""
    a, b = _paired_arrays(x, y)
    res = sps.spearmanr(a, b)
    return CorrelationResult(r=float(res.statistic), p_value=float(res.pvalue))


def kendall_tau(x: object, y: object) -> CorrelationResult:
    """Correlation between equal-length finite samples."""
    a, b = _paired_arrays(x, y)
    res = sps.kendalltau(a, b)
    return CorrelationResult(r=float(res.statistic), p_value=float(res.pvalue))


def chi_square_independence(table: object) -> TestResult:
    """Chi-square test of independence on a valid contingency table."""
    contingency = _contingency_table(table)
    stat, p_value, _dof, _expected = sps.chi2_contingency(contingency)
    return TestResult(float(stat), float(p_value))


def cohens_d(a: object, b: object) -> float:
    """Cohen's d standardized mean difference with pooled sample SD."""
    x, y = _as_1d(a, min_size=2), _as_1d(b, min_size=2)
    nx, ny = x.size, y.size
    pooled_variance = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    if not np.isfinite(pooled_variance) or pooled_variance <= 0.0:
        raise ValueError("cohens_d is undefined when pooled variance is zero")
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled_variance))


def eta_squared_from_f(f_statistic: float, df1: int, df2: int) -> float:
    """Eta-squared effect size derived from a non-negative ANOVA F statistic."""
    if not np.isfinite(f_statistic) or f_statistic < 0.0:
        raise ValueError("f_statistic must be non-negative and finite")
    if df1 <= 0 or df2 <= 0:
        raise ValueError("df1 and df2 must be positive")
    return float((f_statistic * df1) / (f_statistic * df1 + df2))


def cramers_v(table: object) -> float:
    """Cramer's V association strength from a contingency table in [0, 1]."""
    observed = _contingency_table(table)
    stat, _p, _dof, _expected = sps.chi2_contingency(observed)
    n = float(observed.sum())
    denominator = min(observed.shape[0] - 1, observed.shape[1] - 1)
    return float(np.sqrt((stat / n) / denominator))


def percentile(data: object, q: float | list[float]) -> float | list[float]:
    """Percentile(s) of a sample for ``q`` in [0, 100]."""
    values = _as_1d(data)
    q_array = np.asarray(q, dtype=float)
    if not bool(np.all(np.isfinite(q_array))) or np.any((q_array < 0.0) | (q_array > 100.0)):
        raise ValueError("q must contain values in [0, 100]")
    if isinstance(q, list):
        return [float(v) for v in np.percentile(values, q)]
    return float(np.percentile(values, q))


def z_scores(data: object) -> np.ndarray:
    """Standardize a sample to zero mean and unit sample standard deviation."""
    values = _as_1d(data, min_size=2)
    sd = float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        raise ValueError("z_scores undefined for a constant sample")
    return np.asarray((values - np.mean(values)) / sd)


def norm_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal distribution helper with validated location and scale."""
    _validate_normal(mu, sigma)
    if not np.isfinite(x):
        raise ValueError("x must be finite")
    return float(sps.norm.pdf(x, loc=mu, scale=sigma))


def norm_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal distribution helper with validated location and scale."""
    _validate_normal(mu, sigma)
    if not np.isfinite(x):
        raise ValueError("x must be finite")
    return float(sps.norm.cdf(x, loc=mu, scale=sigma))


def norm_ppf(q: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Inverse normal CDF for a probability in [0, 1]."""
    _validate_normal(mu, sigma)
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
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
    """Percentile bootstrap confidence interval for an arbitrary scalar statistic."""
    values = _as_1d(data)
    if (
        not isinstance(n_resamples, (int, np.integer))
        or isinstance(n_resamples, bool)
        or n_resamples < 2
    ):
        raise ValueError("n_resamples must be an integer >= 2")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    stat_fn = statistic if statistic is not None else np.mean
    n = values.size
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n, size=(n_resamples, n))
    samples = values[resample_indices]
    try:
        estimates = np.asarray(stat_fn(samples, axis=1), dtype=float)
    except TypeError:
        estimates = np.array([float(stat_fn(sample)) for sample in samples], dtype=float)
    if estimates.shape != (n_resamples,) or not bool(np.all(np.isfinite(estimates))):
        raise ValueError("statistic must return one finite scalar per resample")
    point_estimate = float(stat_fn(values))
    if not np.isfinite(point_estimate):
        raise ValueError("statistic returned a non-finite point estimate")
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
    if (
        not isinstance(n_permutations, (int, np.integer))
        or isinstance(n_permutations, bool)
        or n_permutations < 1
    ):
        raise ValueError("n_permutations must be a positive integer")
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


def covariance_matrix(data: object) -> np.ndarray:
    """Sample covariance matrix of column-variables (rows = observations)."""
    values = _matrix_observations(data)
    return np.asarray(np.cov(values, rowvar=False, ddof=1))


def correlation_matrix(data: object) -> np.ndarray:
    """Pearson correlation matrix of non-constant column-variables."""
    values = _matrix_observations(data)
    if np.any(np.std(values, axis=0) == 0.0):
        raise ValueError("correlation is undefined for constant variables")
    return np.asarray(np.corrcoef(values, rowvar=False))


def multivariate_normal_logpdf(x: object, mean: object, cov: object) -> float:
    """Log-density of a multivariate normal at a single point."""
    from scipy import stats as sps

    value = sps.multivariate_normal.logpdf(
        np.asarray(x, dtype=float),
        mean=np.asarray(mean, dtype=float),
        cov=np.asarray(cov, dtype=float),
    )
    return float(value)


class StreamingStats:
    """Welford incremental mean/variance for datasets larger than memory."""

    def __init__(self) -> None:
        self.count_value: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0

    def push(self, chunk: object) -> StreamingStats:
        batch = np.asarray(chunk, dtype=float).ravel()
        if batch.size == 0:
            return self
        if not bool(np.all(np.isfinite(batch))):
            raise ValueError("chunk must contain only finite values")
        batch_count = batch.size
        batch_mean = float(batch.mean())
        batch_sum_sq = float(((batch - batch_mean) ** 2).sum())
        delta = batch_mean - self._mean
        total = self.count_value + batch_count
        self._mean += delta * batch_count / total
        self._m2 += batch_sum_sq + delta * delta * self.count_value * batch_count / total
        self.count_value = total
        return self

    def merge(self, other: StreamingStats) -> StreamingStats:
        merged = StreamingStats()
        merged.count_value = self.count_value
        merged._mean = self._mean
        merged._m2 = self._m2

        delta = other._mean - merged._mean
        total = merged.count_value + other.count_value
        if total == 0:
            return merged
        if merged.count_value == 0:
            merged._mean = other._mean
            merged._m2 = other._m2
            merged.count_value = other.count_value
            return merged
        merged._mean += delta * other.count_value / total
        merged._m2 += other._m2 + delta * delta * merged.count_value * other.count_value / total
        merged.count_value = total
        return merged

    @property
    def mean(self) -> float:
        if self.count_value == 0:
            raise ValueError("no observations pushed yet")
        return self._mean

    @property
    def variance(self) -> float:
        if self.count_value < 2:
            raise ValueError("variance needs at least two observations")
        return self._m2 / (self.count_value - 1)

    @property
    def standard_deviation(self) -> float:
        return float(np.sqrt(self.variance))
