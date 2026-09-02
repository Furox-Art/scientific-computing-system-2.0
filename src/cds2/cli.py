"""Command-line interface: ``cds2``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

FUNCS: dict[str, Callable[[float], float]] = {
    "sin": lambda x: __import__("math").sin(x),
    "cos": lambda x: __import__("math").cos(x),
    "exp": lambda x: __import__("math").exp(x),
    "x2": lambda x: x * x,
    "unit": lambda _x: 1.0,
}


def _parse_numbers(raw: str) -> list[float]:
    try:
        return [float(part) for part in raw.replace(";", ",").split(",") if part.strip()]
    except ValueError as exc:
        msg = f"could not parse numbers from {raw!r}"
        raise SystemExit(msg) from exc


def cmd_info(_args: argparse.Namespace) -> int:
    import platform

    import matplotlib
    import numpy
    import pandas
    import scipy

    from . import __version__

    print("scientific-computing-system-2.0")
    print(f"  cds2        {__version__}")
    print(f"  python      {platform.python_version()}")
    print(f"  numpy       {numpy.__version__}")
    print(f"  scipy       {scipy.__version__}")
    print(f"  pandas      {pandas.__version__}")
    print(f"  matplotlib  {matplotlib.__version__}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from .stats import describe

    numbers = _parse_numbers(args.numbers)
    if len(numbers) < 2:
        print("error: stats needs at least two numbers", file=sys.stderr)
        return 1
    result = describe(numbers)
    print(f"n         {result.n}")
    print(f"mean      {result.mean:.6g}")
    print(f"std       {result.std:.6g}")
    print(f"min       {result.minimum:.6g}")
    print(f"q25       {result.q25:.6g}")
    print(f"median    {result.median:.6g}")
    print(f"q75       {result.q75:.6g}")
    print(f"max       {result.maximum:.6g}")
    return 0


def cmd_integrate(args: argparse.Namespace) -> int:
    from .integrate import quad

    func = FUNCS.get(args.func)
    if func is None:
        print(
            f"error: unknown function {args.func!r} (choose from sin|cos|exp|x2|unit)",
            file=sys.stderr,
        )
        return 1
    if args.a >= args.b:
        print("error: --a must be less than --b", file=sys.stderr)
        return 1
    result = quad(func, args.a, args.b)
    print(f"value     {result.value:.10f}")
    print(f"error     {result.error:.3e}")
    return 0


def cmd_linsolve(args: argparse.Namespace) -> int:
    import numpy as np

    from .linalg import solve

    matrix_rows = [row for row in args.a.split(";") if row.strip()]
    a_values = [_parse_numbers(row) for row in matrix_rows]
    widths = {len(row) for row in a_values}
    if not a_values or len(widths) != 1 or len(a_values) != next(iter(widths)):
        print("error: --a must be square, rows separated by ';'", file=sys.stderr)
        return 1
    b_values = _parse_numbers(args.b)
    if len(b_values) != len(a_values):
        print("error: --b must have one value for each row in --a", file=sys.stderr)
        return 1
    try:
        solution = solve(np.array(a_values, dtype=float), np.array(b_values, dtype=float))
    except np.linalg.LinAlgError:
        print("error: --a must be non-singular", file=sys.stderr)
        return 1
    print("solution  " + "  ".join(f"{value:.6g}" for value in np.atleast_1d(solution)))
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from .viz import plot_series, save_figure

    numbers = _parse_numbers(args.numbers)
    fig = plot_series(numbers, title=args.title or "cds2 plot")
    if args.file:
        target = save_figure(fig, args.file)
        print(f"saved     {target}")
    else:
        step = max(len(numbers) // 40, 1)
        peak = max(abs(v) for v in numbers) or 1.0
        for index in range(0, len(numbers), step):
            value = numbers[index]
            bar = "#" * max(int(value / peak * 38) + (1 if value >= 0 else 0), 0)
            print(f"{value:>12.4g} |{bar}")
    return 0


def cmd_entropy(args: argparse.Namespace) -> int:
    from .infotheory import entropy

    probabilities = _parse_numbers(args.probabilities)
    total = sum(probabilities)
    if total <= 0:
        print("error: probabilities must sum to a positive value", file=sys.stderr)
        return 1
    normalized = [value / total for value in probabilities]
    print(f"entropy   {entropy(normalized, base=args.base):.6f}")
    return 0


def cmd_units(args: argparse.Namespace) -> int:
    from .scientific import convert_units

    try:
        result = convert_units(args.value, args.from_unit, args.to_unit)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{args.value:g} {args.from_unit} = {result:.10g} {args.to_unit}")
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    import numpy as np

    coefficients = _parse_numbers(args.coeffs)
    if len(coefficients) < 2 or abs(coefficients[0]) < 1e-15:
        print(
            "error: --coeffs needs descending powers with a nonzero leading term",
            file=sys.stderr,
        )
        return 1
    roots = np.roots(coefficients)
    real_roots = sorted(root.real for root in roots if abs(root.imag) < 1e-9)
    complex_roots = [root for root in roots if abs(root.imag) >= 1e-9]
    for root_value in real_roots:
        print(f"real      {root_value:.8g}")
    for root_value in complex_roots:
        print(f"complex   {root_value.real:.8g} {root_value.imag:+.8g}j")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cds2", description="scientific-computing-system-2.0 command line"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="show version information")
    info_parser.set_defaults(handler=cmd_info)

    stats_parser = subparsers.add_parser(
        "stats", help="descriptive statistics for comma-separated numbers"
    )
    stats_parser.add_argument("numbers")
    stats_parser.set_defaults(handler=cmd_stats)

    integrate_parser = subparsers.add_parser(
        "integrate", help="numerically integrate a built-in function"
    )
    integrate_parser.add_argument("func", help="sin|cos|exp|x2|unit")
    integrate_parser.add_argument("--a", type=float, default=0.0)
    integrate_parser.add_argument("--b", type=float, default=1.0)
    integrate_parser.set_defaults(handler=cmd_integrate)

    linsolve_parser = subparsers.add_parser("linsolve", help="solve A x = b")
    linsolve_parser.add_argument("--a", required=True, help='rows separated by ";", e.g. "3,1;1,2"')
    linsolve_parser.add_argument("--b", required=True, help="right-hand side values, e.g. 9,8")
    linsolve_parser.set_defaults(handler=cmd_linsolve)

    plot_parser = subparsers.add_parser("plot", help="plot comma-separated numbers (ASCII or PNG)")
    plot_parser.add_argument("numbers")
    plot_parser.add_argument(
        "--file", default="", help="save PNG to this path instead of ASCII output"
    )
    plot_parser.add_argument("--title", default="")
    plot_parser.set_defaults(handler=cmd_plot)

    entropy_parser = subparsers.add_parser(
        "entropy", help="Shannon entropy of comma-separated probabilities"
    )
    entropy_parser.add_argument("probabilities")
    entropy_parser.add_argument("--base", type=float, default=2.0)
    entropy_parser.set_defaults(handler=cmd_entropy)

    units_parser = subparsers.add_parser("units", help="convert between physical units")
    units_parser.add_argument("value", type=float)
    units_parser.add_argument("--from-unit", required=True, dest="from_unit")
    units_parser.add_argument("--to-unit", required=True, dest="to_unit")
    units_parser.set_defaults(handler=cmd_units)

    solve_parser = subparsers.add_parser(
        "solve", help="roots of a polynomial given in descending powers"
    )
    solve_parser.add_argument(
        "--coeffs", required=True, help='descending powers, e.g. "1,-5,6" for x^2-5x+6'
    )
    solve_parser.set_defaults(handler=cmd_solve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
