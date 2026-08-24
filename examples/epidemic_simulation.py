"""Case study 16: SIR/SEIR epidemic scenarios.

Two outbreak scenarios - an unmitigated respiratory pathogen and a
flattened-curve intervention that reduces beta mid-course via a second
simulation - are compared with herd-immunity and final-size analytics.
"""

from __future__ import annotations

import numpy as np

import cds2


def report(label: str, result: cds2.epidemiology.SIRResult | cds2.epidemiology.SEIRResult) -> None:
    print(f"-- {label}")
    print(
        f"   R0={result.r0:.2f}  peak day={result.peak_day:.0f}  "
        f"peak infected={result.infected.max():,.0f}  attack rate={result.attack_rate:.1%}"
    )


def main() -> None:
    population, days = 1_000_000.0, 120
    beta, gamma = 0.30, 0.10

    print("== Unmitigated SIR outbreak ==")
    wild = cds2.epidemiology.simulate_sir(population, beta=beta, gamma=gamma, days=days, i0=50.0)
    report("wild type", wild)

    mitigated = cds2.epidemiology.simulate_sir(
        population, beta=beta * 0.55, gamma=gamma, days=days, i0=50.0
    )
    report("mitigated (45% contact cut)", mitigated)
    reduction = 100.0 * (1.0 - mitigated.infected.max() / wild.infected.max())
    print(f"   peak reduction          : {reduction:.0f}%")

    print("\n== SEIR with incubation period ==")
    seir = cds2.epidemiology.simulate_seir(
        population, beta=beta, sigma=0.20, gamma=gamma, days=days, i0=50.0, e0=100.0
    )
    report("SEIR (5-day incubation)", seir)
    lag = int(np.argmax(seir.exposed)) - int(np.argmax(seir.infected[: len(seir.exposed)]))
    print(f"   exposed curve leads infections by {abs(lag)} days")

    hit = cds2.epidemiology.herd_immunity_threshold(wild.r0)
    final_size = cds2.epidemiology.final_size_iteration(wild.r0)
    print("\n== Analytical cross-checks ==")
    print(f"herd immunity threshold : {hit:.1%}")
    print(f"final size (analytic)   : {final_size:.1%}")
    print(f"final size (simulated)  : {wild.attack_rate:.1%}")


if __name__ == "__main__":
    main()
