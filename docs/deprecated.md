# Deprecated modules

Two cds2 modules are deprecated since **4.3.0** and may be removed in a
future major release. Both still work, but importing them emits a
`DeprecationWarning` — including on a plain `import cds2`, because the
package root re-exports every module:

```text
DeprecationWarning: cds2.distributions is deprecated since 4.3.0 ...
DeprecationWarning: cds2.special is deprecated since 4.3.0 ...
```

## `cds2.special` → `scipy.special`

The runtime message, quoted verbatim from `src/cds2/special.py`:

> cds2.special is deprecated since 4.3.0 and may be removed in a future
> major release: it is a thin convenience wrapper around scipy.special
> that coerces scalars to atleast_1d and lags upstream releases. Use
> `from scipy import special as sps` directly (e.g. sps.gamma, sps.erf,
> sps.jv).

Migrate by replacing `cds2.special.<fn>` with the same-named
`scipy.special` function, for example:

```python
from scipy import special as sps

sps.gamma(5.0)
sps.erf(1.0)
```

API page (kept for the transition period): [cds2.special](api/special.md).

## `cds2.distributions` → `scipy.stats`

The runtime message, quoted verbatim from `src/cds2/distributions.py`:

> cds2.distributions is deprecated since 4.3.0 and may be removed in a
> future major release: it re-exports scipy.stats distribution methods as
> pdf/cdf/ppf aliases with divergent parameter names and without the
> frozen-distribution, rvs, fit, or interval API. Use `from scipy import
> stats` directly (e.g. stats.norm.pdf, stats.t.cdf, stats.binom.ppf) or
> frozen `stats.norm(loc=mu, scale=sigma)`.

Migrate by calling `scipy.stats` distribution methods directly, for example:

```python
from scipy import stats

stats.norm.pdf(0.0)
stats.t.cdf(1.5, df=10)
```

API page (kept for the transition period):
[cds2.distributions](api/distributions.md).
