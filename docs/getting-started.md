# Getting started

## Installation

CDS v2 requires Python 3.10+ and installs its scientific dependencies
automatically:

```bash
pip install scientific-computing-system-2.0
```

Development install from source:

```bash
git clone https://github.com/Furox-Art/scientific-computing-system-2.0.git
cd scientific-computing-system-2.0
pip install -e .[dev]
```

## Ten-minute tour

### Linear algebra

```python
import numpy as np
from cds2 import linalg

a = [[4.0, 7.0], [2.0, 6.0]]
print(linalg.det(a))  # 10.0
print(linalg.solve(a, [18.0, 16.0]))  # [2.5, 1.0]

svd = linalg.svd(np.random.default_rng(0).normal(size=(5, 3)))
print(svd.s)  # descending singular values
```

### Statistics

```python
from cds2 import stats

treatment = [12.9, 13.5, 12.8, 15.6, 17.2, 19.2]
control = [12.7, 13.6, 12.0, 15.2, 16.8, 20.0]

result = stats.independent_t_test(treatment, control)
print(result.statistic, result.p_value)

summary = stats.describe(treatment)
print(summary.mean, summary.std)
```

### Optimization

```python
from cds2 import optimize

rosen = lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2
res = optimize.minimize(rosen, x0=[0.0, 0.0])
print(res.x)  # ~ [1.0, 1.0]

root = optimize.find_root_scalar(lambda x: x * x - 4, 0.0, 10.0)
print(root)  # 2.0
```

### ODEs and integration

```python
import numpy as np
from cds2 import integrate

growth = integrate.solve_ivp(lambda t, y: 0.5 * y, (0.0, 10.0), [1.0])
print(growth.y[0][-1])  # ~ e^5

area = integrate.quad(np.sin, 0.0, np.pi)
print(area.value)  # 2.0
```

### Signals

```python
import numpy as np
from cds2 import signals

fs = 1000.0
t = np.arange(4000) / fs
tone = np.sin(2 * np.pi * 50.0 * t)

freqs, psd = signals.power_spectrum(tone, fs=fs)
peak = freqs[np.argmax(psd)]
print(peak)  # ~ 50 Hz
```

### Graphs

```python
from cds2 import graph

edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
adj = graph.from_edges(4, edges, directed=True)

print(graph.single_source_shortest_paths(adj, source=0))
print(graph.pagerank(adj))
print(graph.topological_order(4, edges))
```

### Machine learning

```python
from cds2 import ml

x, y = ml.make_regression_data(n=300, n_features=3, noise=5.0, seed=1)
x_train, x_test, y_train, y_test = ml.train_test_split(x, y, seed=2)

model = ml.LinearRegression().fit(x_train, y_train)
print(model.score(x_test, y_test))  # close to 1.0
```

### Time series

```python
import numpy as np
import pandas as pd
from cds2 import timeseries

t_values = np.arange(120, dtype=float)
series = pd.Series(0.3 * t_values + 4 * np.sin(2 * np.pi * t_values / 12))

decomposition = timeseries.seasonal_decompose(series, period=12)
smoothed = timeseries.moving_average(series, window=5)
test = timeseries.ljung_box(series.diff().dropna(), lags=10)
```

### Visualization and I/O

```python
import pandas as pd
from cds2 import io, viz

viz.plot_series([1, 3, 2, 5, 4], title="demo", save="series.png")
viz.plot_histogram([1, 2, 2, 3, 3, 3, 4, 5])

frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
io.write_csv(frame, "data.csv")
loaded = io.read_csv("data.csv")
print(io.summarize(loaded))
```

### Information theory

```python
import numpy as np
from cds2 import infotheory

print(infotheory.entropy([0.25, 0.25, 0.25, 0.25]))  # 2.0 bits
print(infotheory.mutual_information([[0.5, 0.0], [0.0, 0.5]]))  # 1.0 bit
print(infotheory.permutation_entropy(np.sin(np.linspace(0, 50, 500))))
```

### Chaos and nonlinear dynamics

```python
from cds2 import chaos

series = chaos.logistic_map(3.99, length=600, seed=1)
print(chaos.largest_lyapunov_exponent(series).exponent)  # > 0 for chaotic r
print(chaos.correlation_dimension(series).dimension)
print(chaos.hurst_exponent(np.cumsum(np.random.default_rng(0).normal(size=2000))))
```

### Bayesian inference

```python
from cds2 import bayes

posterior = bayes.beta_binomial_update(successes=7, failures=3)
print(posterior.mean)  # 0.7

classifier = bayes.NaiveBayes().fit([[0, 0], [0, 1], [10, 10], [10, 11]], [0, 0, 1, 1])
print(classifier.predict([[0, 0], [10, 10]]))  # [0. 1.]
```

### Metaheuristics

```python
from cds2 import metaheuristics

result = metaheuristics.pso_minimize(
    lambda v: (v[0] - 3) ** 2 + (v[1] + 2) ** 2,
    [(-10, 10), (-10, 10)],
    seed=1,
)
print(result.x, result.fun)
```

### Geometry

```python
from cds2 import geometry

print(geometry.hull_area([(0, 0), (1, 0), (0, 1)]))  # 0.5
print(geometry.point_in_polygon((0.2, 0.2), [(0, 0), (1, 0), (1, 1), (0, 1)]))
print(geometry.closest_pair([(0, 0), (5, 5), (1, 1), (9, 9)]))
```

### Reinforcement learning

```python
from cds2 import rl

bandit_result = rl.ucb1(rl.Bandit([0.2, 0.8], seed=1), episodes=1000)
print(bandit_result.counts)  # arm 1 dominates

q_values, returns = rl.q_learn(rl.GridWorld(4, 4), episodes=400, seed=1)
print(returns[-30:].mean())  # ~ 1.0 once the policy is learned
```

## Command line

```text
cds2 info                       version table for the whole stack
cds2 stats 1,2,3,4,5            descriptive statistics
cds2 integrate sin --a 0 --b 3.14159
cds2 linsolve --a "3,1;1,2" --b "9,8"
cds2 entropy "0.25,0.25,0.25,0.25"
cds2 units 5 --from-unit km --to-unit mile
cds2 solve --coeffs "1,-5,6"
cds2 plot 1,3,2,5,4 --file out.png
```

## Relationship to CDS v1.x

The original zero-dependency line remains at
[Furox88/cognitive-discovery-system](https://github.com/Furox88/cognitive-discovery-system).
v2 intentionally trades that constraint for the speed of the scientific stack;
the module layout stays familiar so migration is mostly changing `cds.` to
`cds2.`.
