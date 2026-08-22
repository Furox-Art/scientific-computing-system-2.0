"""Case study 3: Bayesian inference of a sensor bias with MCMC.

Ten calibration readings are modelled as Normal(true_offset, known_sigma).
A flat prior on the offset is sampled with Metropolis-Hastings; the
posterior mean and credible interval are compared against the analytic
conjugate answer.
"""

from __future__ import annotations

import numpy as np

import cds2

KNOWN_SIGMA = 0.8
READINGS = np.array([1.21, 0.94, 1.07, 1.38, 0.88, 1.12, 1.02, 0.96, 1.19, 1.05])


def log_posterior(offset: np.ndarray) -> float:
    likelihood = -0.5 * float(np.sum(((READINGS - offset[0]) / KNOWN_SIGMA) ** 2))
    return likelihood


def main() -> None:
    chain = cds2.montecarlo.metropolis_hastings(
        log_posterior,
        initial=[0.0],
        n_samples=30_000,
        burn_in=3_000,
        proposal_scale=0.25,
        seed=2024,
    )
    samples = chain.samples[:, 0]

    posterior_mean = float(np.mean(samples))
    credible = cds2.stats.percentile(samples.tolist(), [2.5, 97.5])

    analytic_mean = float(np.mean(READINGS))
    analytic_se = KNOWN_SIGMA / np.sqrt(READINGS.size)

    print("== Bayesian sensor-bias inference ==")
    print(f"MCMC     posterior mean : {posterior_mean:.4f}")
    print(f"Analytic posterior mean  : {analytic_mean:.4f}")
    print(f"95% credible interval    : [{credible[0]:.4f}, {credible[1]:.4f}]")
    print(
        f"Analytic interval        : "
        f"[{analytic_mean - 1.96 * analytic_se:.4f}, {analytic_mean + 1.96 * analytic_se:.4f}]"
    )
    print(f"Acceptance rate          : {chain.acceptance_rate:.2f}")


if __name__ == "__main__":
    main()
