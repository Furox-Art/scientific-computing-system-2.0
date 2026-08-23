"""Case study 9: Conjugate Bayesian A/B analysis.

Two website variants are tested. Beta-Binomial conjugate updates give full
posteriors for each conversion rate; Monte Carlo draws over those
posteriors estimate the probability that B beats A and a credible interval
on the uplift - all without MCMC.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

import cds2

VISITORS_A, CONVERSIONS_A = 1_200, 132
VISITORS_B, CONVERSIONS_B = 1_180, 156


def main() -> None:
    posterior_a = cds2.bayes.beta_binomial_update(
        successes=CONVERSIONS_A, failures=VISITORS_A - CONVERSIONS_A
    )
    posterior_b = cds2.bayes.beta_binomial_update(
        successes=CONVERSIONS_B, failures=VISITORS_B - CONVERSIONS_B
    )

    print("== A/B conversion posteriors ==")
    for label, post in (("A", posterior_a), ("B", posterior_b)):
        low, high = cds2.bayes.posterior_interval(sp_stats.beta(post.alpha, post.beta))
        print(
            f"variant {label}: mean={post.mean:.4f}  mode={post.mode:.4f}  95% CI=[{low:.4f}, {high:.4f}]"
        )

    rng = np.random.default_rng(19)
    draws_a = sp_stats.beta(posterior_a.alpha, posterior_a.beta).rvs(size=200_000, random_state=rng)
    draws_b = sp_stats.beta(posterior_b.alpha, posterior_b.beta).rvs(size=200_000, random_state=rng)
    probability = float(np.mean(draws_b > draws_a))
    uplift = draws_b / draws_a - 1.0
    lo, hi = cds2.stats.percentile(uplift.tolist(), [2.5, 97.5])

    print(f"P(B > A)                 : {probability:.3f}")
    print(f"95% CI on relative uplift: [{lo:+.3f}, {hi:+.3f}]")

    classifier = cds2.bayes.NaiveBayes().fit(
        [[0.0, 0.0], [0.2, 0.1], [0.1, 0.0], [3.0, 3.2], [2.8, 3.0], [3.1, 2.9]],
        [0, 0, 0, 1, 1, 1],
    )
    print(f"naive Bayes sanity check : {classifier.predict([[0.1, 0.05], [3.0, 3.1]])}")


if __name__ == "__main__":
    main()
