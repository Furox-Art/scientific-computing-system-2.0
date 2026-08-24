"""Case study 14: Component reliability from censored field data.

A fleet of 250 components is tracked; lifetimes come from a Weibull
process and a third of the records are right-censored. Kaplan-Meier
estimates the survival curve, Weibull fitting recovers the generating
parameters, and the bathtub model sketches the hazard profile.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

import cds2

TRUE_SHAPE, TRUE_SCALE, N_COMPONENTS = 1.8, 850.0, 250
CENSOR_FRACTION = 0.3


def main() -> None:
    rng = np.random.default_rng(7)
    durations = sp_stats.weibull_min.rvs(
        TRUE_SHAPE, scale=TRUE_SCALE, size=N_COMPONENTS, random_state=rng
    )
    censor_mask = rng.random(N_COMPONENTS) < CENSOR_FRACTION
    durations[censor_mask] = rng.uniform(50.0, 400.0, size=int(censor_mask.sum()))
    events = (~censor_mask).astype(int)

    km = cds2.reliability.kaplan_meier(durations.tolist(), events.tolist())
    print("== Kaplan-Meier estimate ==")
    print(
        f"components : {N_COMPONENTS} ({int(events.sum())} failures, {int(censor_mask.sum())} censored)"
    )
    median = km.median
    print(f"median life: {median:.0f} h" if median is not None else "median life: not reached")

    fit = cds2.reliability.weibull_fit(durations.tolist(), failures_mask=(events == 1).tolist())
    print("\n== Weibull fit (failures only) ==")
    print(f"shape : true {TRUE_SHAPE:.2f} vs fitted {fit.shape:.2f}")
    print(f"scale : true {TRUE_SCALE:.0f} h vs fitted {fit.scale:.0f} h")

    print("\n== Fitted survival curve ==")
    grid = [200, 400, 600, 800, 1000]
    curve = cds2.reliability.weibull_survival(grid, fit.shape, fit.scale)
    for t_value, s_value in zip(grid, curve, strict=True):
        bar = "#" * int(s_value * 40)
        print(f"{t_value:>5d} h  S={s_value:.3f} |{bar}")

    operating_time = float(durations.sum())
    failure_count = int(events.sum())
    mtbf = cds2.reliability.mtbf(operating_time, failure_count)
    availability = cds2.reliability.availability(mtbf, mttr=48.0)
    print("\n== Fleet metrics ==")
    print(f"MTBF       : {mtbf:.0f} h")
    print("MTTR       : 48 h")
    print(f"availability: {availability:.4f}")

    hazard = cds2.reliability.bathtub_curve(
        [10.0, 100.0, 500.0, 2000.0, 5000.0],
        early_rate=0.05,
        intrinsic_rate=0.002,
        wearout_rate=0.02,
        knee_early=80.0,
        knee_wearout=1500.0,
    )
    print("\n== Bathtub hazard profile ==")
    for t_value, h_value in zip([10, 100, 500, 2000, 5000], hazard, strict=True):
        print(f"{t_value:>5d} h  lambda={h_value:.5f}/h")


if __name__ == "__main__":
    main()
