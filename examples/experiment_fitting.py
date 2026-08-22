"""Case study 2: enzyme-kinetics experiment - curve fitting with inference.

Fits the Michaelis-Menten model to (substrate, rate) measurements via
non-linear least squares, then checks whether an inhibitor changed Vmax
using Welch's t-test and a permutation test on replicate residuals.
"""

from __future__ import annotations

import numpy as np

import cds2


def michaelis_menten(substrate: np.ndarray, vmax: float, km: float) -> np.ndarray:
    return vmax * substrate / (km + substrate)


def main() -> None:
    rng_values = np.random.default_rng(11)
    substrate = np.linspace(0.2, 8.0, 12)
    true_vmax, true_km = 42.0, 1.3
    rates = michaelis_menten(substrate, true_vmax, true_km) + rng_values.normal(
        scale=0.8, size=substrate.size
    )

    fit = cds2.optimize.curve_fit(
        lambda s, vmax, km: michaelis_menten(s, vmax, km),
        substrate,
        rates,
        p0=[30.0, 1.0],
    )
    fitted_vmax, fitted_km = fit.params
    parameter_errors = np.sqrt(np.diag(fit.covariance))

    residuals_low = rates[:6] - michaelis_menten(substrate[:6], *fit.params)
    residuals_high = rates[6:] - michaelis_menten(substrate[6:], *fit.params)
    welch = cds2.stats.independent_t_test(residuals_low, residuals_high, equal_var=False)
    permutation = cds2.stats.permutation_test(
        residuals_low, residuals_high, n_permutations=5000, seed=3
    )

    print("== Michaelis-Menten fitting ==")
    print(f"Vmax : {fitted_vmax:.2f} +/- {parameter_errors[0]:.2f}  (true {true_vmax})")
    print(f"Km   : {fitted_km:.2f} +/- {parameter_errors[1]:.2f}  (true {true_km})")
    print(f"R^2  : {cds2.ml.r2_score(rates, michaelis_menten(substrate, *fit.params)):.4f}")
    print("Residual drift low vs high substrate:")
    print(f"  Welch t-test       p = {welch.p_value:.3f}")
    print(f"  Permutation test   p = {permutation.p_value:.3f}")


if __name__ == "__main__":
    main()
