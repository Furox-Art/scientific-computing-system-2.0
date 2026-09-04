# cds2.optimize

Optimization and root finding on scipy.optimize.

## Curve-fitting diagnostics

`curve_fit` supports measurement uncertainty (`sigma` / `absolute_sigma`),
parameter bounds, solver selection and optional Jacobians. In addition to fitted
parameters and covariance, `FitResult` reports parameter standard errors,
predictions, residuals, RSS, RMSE, R² and residual degrees of freedom.

The residual diagnostics are calculated on the original, unweighted residuals;
`sigma` controls the parameter fit and covariance estimation.

```python
import numpy as np

from cds2 import optimize

x = np.linspace(0.0, 4.0, 20)
y = 2.5 * x + 1.2
sigma = np.full_like(x, 0.2)

fit = optimize.curve_fit(
    lambda t, slope, intercept: slope * t + intercept,
    x,
    y,
    p0=[1.0, 0.0],
    sigma=sigma,
    absolute_sigma=True,
    bounds=([0.0, -5.0], [5.0, 5.0]),
)

print(fit.params)
print(fit.parameter_std)
print(fit.rmse, fit.r_squared)
```

::: cds2.optimize
