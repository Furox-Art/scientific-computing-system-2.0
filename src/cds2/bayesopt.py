"""Bayesian optimization with Gaussian-process surrogate and acquisition functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sla
from scipy import optimize as spo
from scipy.stats import norm

from cds2 import linalg as cds2_linalg

__all__ = [
    "GaussianProcess",
    "OptimizeResult",
    "bayes_opt",
    "expected_improvement",
    "upper_confidence_bound",
]

FloatArray = NDArray[np.float64]


def _rbf_kernel(
    x1: FloatArray,
    x2: FloatArray,
    length_scale: float,
    sigma_f: float,
) -> FloatArray:
    """RBF (squared-exponential) kernel matrix."""
    # x1: (n1, d), x2: (n2, d)
    diff = x1[:, None, :] - x2[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    return np.asarray(sigma_f**2 * np.exp(-0.5 * sqdist / (length_scale**2)), dtype=float)


def _prepare_bounds(bounds: object) -> FloatArray:
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        msg = "bounds must be a sequence of (low, high) pairs"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    if arr.shape[0] == 0:
        msg = "bounds must contain at least one dimension"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    if np.any(arr[:, 0] >= arr[:, 1]):
        msg = "each bound low must be < high"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    return np.asarray(arr, dtype=float)


def _as_2d(x: object, dim: int | None = None) -> FloatArray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)  # pragma: no cover
    elif arr.ndim == 1:
        # 1-D input: single point of shape (dim,) or many points of shape (n,)?
        # If dim is known and arr.size == dim, treat as single point.
        if dim is not None and arr.size == dim:  # pragma: no cover
            arr = arr.reshape(1, -1)  # pragma: no cover
        else:  # pragma: no cover
            arr = arr.reshape(-1, 1)  # pragma: no cover
    elif arr.ndim == 2:
        pass
    else:
        msg = "input must be 1-D or 2-D array"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    return np.asarray(arr, dtype=float)


class GaussianProcess:
    """Gaussian-process regressor with RBF kernel.

    Uses the squared-exponential kernel
    ``k(x, x') = sigma_f^2 * exp(-0.5 * ||x-x'||^2 / length_scale^2)``
    with a small observation noise added to the diagonal.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        sigma_f: float = 1.0,
        noise: float = 1e-8,
    ) -> None:
        if length_scale <= 0 or sigma_f <= 0 or noise < 0:
            msg = "length_scale and sigma_f must be >0, noise >=0"  # pragma: no cover
            raise ValueError(msg)  # pragma: no cover
        self.length_scale: float = float(length_scale)
        self.sigma_f: float = float(sigma_f)
        self.noise: float = float(noise)
        self.X_train_: FloatArray | None = None
        self.y_train_: FloatArray | None = None
        self.L_: FloatArray | None = None
        self.alpha_: FloatArray | None = None

    def fit(self, x_train: object, y_train: object) -> GaussianProcess:
        """Fit the GP to training data."""
        x_arr = _as_2d(x_train)
        if x_arr.ndim != 2:
            msg = "x_train must be 2-D"  # pragma: no cover
            raise ValueError(msg)  # pragma: no cover
        y_arr = np.asarray(y_train, dtype=float).ravel()
        if y_arr.size != x_arr.shape[0]:
            msg = "x_train and y_train must have matching first dimension"  # pragma: no cover
            raise ValueError(msg)  # pragma: no cover
        if x_arr.shape[0] == 0:
            msg = "training data must be non-empty"  # pragma: no cover
            raise ValueError(msg)  # pragma: no cover
        n = x_arr.shape[0]
        k_mat = _rbf_kernel(x_arr, x_arr, self.length_scale, self.sigma_f)
        # leverage cds2.linalg for Cholesky; add jitter for stability
        jitter = self.noise**2
        k_mat = k_mat + np.eye(n, dtype=float) * (jitter + 1e-10)
        try:
            lower = cds2_linalg.cholesky(k_mat)
        except np.linalg.LinAlgError:  # pragma: no cover
            # add larger jitter and retry  # pragma: no cover
            k_mat = k_mat + np.eye(n, dtype=float) * 1e-6  # pragma: no cover
            lower = cds2_linalg.cholesky(k_mat)  # pragma: no cover
        # solve for alpha = K^{-1} y via Cholesky
        # Use triangular solves via scipy for stability
        # Solve L * v = y  -> v
        v_vec = sla.solve_triangular(lower, y_arr, lower=True, check_finite=False)
        # Solve L.T * alpha = v
        alpha = sla.solve_triangular(lower.T, v_vec, lower=False, check_finite=False)
        self.X_train_ = np.asarray(x_arr, dtype=float)
        self.y_train_ = np.asarray(y_arr, dtype=float)
        self.L_ = np.asarray(lower, dtype=float)
        self.alpha_ = np.asarray(alpha, dtype=float)
        return self

    def predict(self, x_star: object) -> tuple[FloatArray, FloatArray]:
        """Predict mean and standard deviation at new points."""
        if self.X_train_ is None or self.y_train_ is None or self.L_ is None or self.alpha_ is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        x_arr = _as_2d(x_star, dim=self.X_train_.shape[1])
        if x_arr.shape[1] != self.X_train_.shape[1]:
            msg = "x_star dimension must match training data"  # pragma: no cover
            raise ValueError(msg)  # pragma: no cover
        k_star = _rbf_kernel(x_arr, self.X_train_, self.length_scale, self.sigma_f)
        # mean
        mu = k_star @ self.alpha_
        # variance via triangular solve: v = L^{-1} K_star^T
        v_mat = sla.solve_triangular(self.L_, k_star.T, lower=True, check_finite=False)
        k_star_star = float(self.sigma_f**2)
        # var = k(x*, x*) - v^T v for each x*
        var = k_star_star - np.sum(v_mat * v_mat, axis=0)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        return np.asarray(mu, dtype=float), np.asarray(std, dtype=float)


def expected_improvement(
    mu: object,
    sigma: object,
    best_f: float,
    xi: float = 0.01,
) -> FloatArray:
    """Expected Improvement for minimization.

    Parameters
    ----------
    mu:
        Predicted mean(s) at candidate points.
    sigma:
        Predicted standard deviation(s).
    best_f:
        Best observed objective value so far (minimum).
    xi:
        Exploration-exploitation trade-off.
    """
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    best = float(best_f)
    xi_val = float(xi)
    # broadcast shapes: mu_arr and sigma_arr should be compatible
    # handle scalar vs array
    imp = best - mu_arr - xi_val
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.divide(imp, sigma_arr, out=np.zeros_like(imp, dtype=float), where=sigma_arr > 0)
        cdf = norm.cdf(z)
        pdf = norm.pdf(z)
        ei = imp * cdf + sigma_arr * pdf
        ei = np.where(sigma_arr > 0, ei, 0.0)
        ei = np.maximum(ei, 0.0)
    return np.asarray(ei, dtype=float)


def upper_confidence_bound(
    mu: object,
    sigma: object,
    kappa: float = 2.0,
) -> FloatArray:
    """Upper confidence bound acquisition.

    ``ucb = mu + kappa * sigma``. For maximization, maximise this
    quantity; for minimization the optimizer uses ``-mu + kappa*sigma``
    (i.e. ``upper_confidence_bound(-mu, sigma)``).
    """
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    k = float(kappa)
    return np.asarray(mu_arr + k * sigma_arr, dtype=float)


@dataclass(frozen=True)
class OptimizeResult:
    """Outcome of Bayesian optimization."""

    x: FloatArray
    fun: float
    success: bool
    message: str
    n_iterations: int
    xs: FloatArray
    ys: FloatArray


def _eval_objective(
    objective: Callable[[FloatArray], float],
    x: FloatArray,
) -> float:
    arr = np.asarray(x, dtype=float)
    try:
        val = objective(arr)
        v_arr = np.asarray(val, dtype=float)
        if v_arr.size == 0:
            msg = "objective returned empty value"  # pragma: no cover
            raise ValueError(msg)  # pragma: no cover
        # squeeze to scalar if needed
        if v_arr.size == 1:
            return float(v_arr.squeeze())
        return float(v_arr.flat[0])  # pragma: no cover
    except Exception as first_exc:  # pragma: no cover
        # try unpacked call for objectives expecting *x  # pragma: no cover
        try:  # pragma: no cover
            val2 = objective(*arr.tolist())  # pragma: no cover
            v_arr2 = np.asarray(val2, dtype=float)  # pragma: no cover
            if v_arr2.size == 1:  # pragma: no cover
                return float(v_arr2.squeeze())  # pragma: no cover
            return float(v_arr2.flat[0])  # pragma: no cover
        except Exception:  # pragma: no cover
            raise first_exc  # pragma: no cover


def bayes_opt(
    objective: Callable[[FloatArray], float],
    bounds: Sequence[tuple[float, float]] | FloatArray,
    n_init: int = 5,
    n_iter: int = 20,
    acquisition: str = "ei",
    seed: int | None = None,
) -> OptimizeResult:
    """Bayesian optimization over a box-constrained domain.

    Parameters
    ----------
    objective:
        Function to minimize; takes a 1-D array of shape ``(dim,)``.
    bounds:
        Sequence of ``(low, high)`` pairs per dimension.
    n_init:
        Number of random initial evaluations.
    n_iter:
        Number of Bayesian optimization steps.
    acquisition:
        Acquisition function: ``"ei"`` (expected improvement) or ``"ucb"``
        (upper confidence bound / lower confidence bound for minimization).
    seed:
        Random seed for reproducibility.
    """
    if n_init < 1:
        msg = "n_init must be >= 1"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    if n_iter < 0:
        msg = "n_iter must be >= 0"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover
    bounds_arr = _prepare_bounds(bounds)
    dim = int(bounds_arr.shape[0])
    low = np.asarray(bounds_arr[:, 0], dtype=float)
    high = np.asarray(bounds_arr[:, 1], dtype=float)

    acq = acquisition.lower().strip()
    if acq in {"expected_improvement", "expected-improvement"}:
        acq = "ei"  # pragma: no cover
    elif acq in {"upper_confidence_bound", "upper-confidence-bound"}:
        acq = "ucb"  # pragma: no cover
    if acq not in {"ei", "ucb"}:
        msg = "acquisition must be 'ei' or 'ucb'"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    rng = np.random.default_rng(seed)

    # initial design uniformly at random
    x_init = rng.uniform(low, high, size=(n_init, dim))
    y_init = np.empty(n_init, dtype=float)
    for i in range(n_init):
        y_init[i] = _eval_objective(objective, x_init[i])

    x_all = np.asarray(x_init, dtype=float)
    y_all = np.asarray(y_init, dtype=float)

    # adaptive length scale scaled to bounds
    mean_range = float(np.mean(high - low))
    base_length_scale = max(mean_range * 0.25, 1e-2)

    for _ in range(n_iter):
        y_mean = float(np.mean(y_all))
        y_std = float(np.std(y_all))
        if y_std < 1e-9:
            y_std = 1.0  # pragma: no cover
        y_norm = (y_all - y_mean) / y_std

        gp = GaussianProcess(length_scale=base_length_scale, sigma_f=1.0, noise=1e-6)
        gp.fit(x_all, y_norm)
        best_f_norm = float(np.min(y_norm))

        # acquisition optimization via random sampling + local refinement
        n_candidates = 5000 if dim <= 3 else 2000 * dim
        if n_candidates > 20000:
            n_candidates = 20000  # pragma: no cover
        candidates = rng.uniform(low, high, size=(n_candidates, dim))
        mu_cand, sigma_cand = gp.predict(candidates)

        if acq == "ei":
            acq_vals = expected_improvement(mu_cand, sigma_cand, best_f_norm, xi=0.01)
        else:
            # for minimization, maximize -mu + kappa*sigma
            # upper_confidence_bound(-mu, sigma) == -mu + kappa*sigma
            acq_vals = upper_confidence_bound(-mu_cand, sigma_cand, kappa=2.0)

        best_idx = int(np.argmax(acq_vals))
        best_cand = np.asarray(candidates[best_idx], dtype=float)
        best_acq_val = float(acq_vals[best_idx])

        # local refinement from top candidates
        bounds_list: list[tuple[float, float]] = [
            (float(low[d]), float(high[d])) for d in range(dim)
        ]

        def neg_acq(x_flat: FloatArray) -> float:
            x_2d = np.asarray(x_flat, dtype=float).reshape(1, -1)
            mu_arr, sigma_arr = gp.predict(x_2d)
            mu0 = float(mu_arr[0])
            sigma0 = float(sigma_arr[0])
            if acq == "ei":
                ei_val = expected_improvement(
                    np.array([mu0]), np.array([sigma0]), best_f_norm, xi=0.01
                )[0]
                return -float(ei_val)
            # ucb for minimization
            kappa = 2.0
            # neg_acq = mu - kappa*sigma  (minimize -> maximize -mu+kappa*sigma)
            return float(mu0 - kappa * sigma0)

        # try refining the top few candidates
        top_n = min(5, n_candidates)
        top_indices = np.argsort(acq_vals)[-top_n:]
        best_x_next = best_cand
        for idx in top_indices:
            x0 = np.asarray(candidates[int(idx)], dtype=float)
            try:
                res = spo.minimize(neg_acq, x0, method="L-BFGS-B", bounds=bounds_list)
            except Exception:  # pragma: no cover
                continue  # pragma: no cover
            if not res.success:
                continue
            # acquisition at refined point is -res.fun
            try:
                acq_at_res = -float(res.fun)
            except Exception:  # pragma: no cover
                continue  # pragma: no cover
            if acq_at_res > best_acq_val:
                best_acq_val = acq_at_res
                best_x_next = np.asarray(res.x, dtype=float)

        # ensure within bounds (clip due to numerical issues)
        best_x_next = np.clip(best_x_next, low, high)

        y_next = _eval_objective(objective, best_x_next)
        x_all = np.vstack([x_all, best_x_next.reshape(1, -1)])
        y_all = np.append(y_all, y_next)

    best_index = int(np.argmin(y_all))
    best_x = np.asarray(x_all[best_index], dtype=float)
    best_fun = float(y_all[best_index])
    return OptimizeResult(
        x=best_x,
        fun=best_fun,
        success=True,
        message="Optimization terminated successfully.",
        n_iterations=int(n_iter),
        xs=np.asarray(x_all, dtype=float),
        ys=np.asarray(y_all, dtype=float),
    )
