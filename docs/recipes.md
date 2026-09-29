# Common Recipes

Copy-paste solutions for things people actually do.

## Fit a curve to experimental data

```python
from cds2 import modeling

result = modeling.fit(x_data, y_data, model="exponential")
print(result.params, result.r_squared)
```

## Run a Monte Carlo risk analysis

```python
from cds2 import montecarlo

outcomes = montecarlo.simulate(portfolio_returns, n=10000)
print(outcomes.var_95)  # Value at Risk
```

## Bayesian A/B test

```python
from cds2 import bayes

result = bayes.ab_test(
    control=1000, control_conversions=50, treatment=1000, treatment_conversions=65
)
print(result.probability_better)  # P(treatment
```

## Solve a stiff ODE

```python
from cds2 import sde

sol = sde.solve_stiff(reaction_kinetics, t_span=(0, 100), y0=initial)
sol.plot()
```
