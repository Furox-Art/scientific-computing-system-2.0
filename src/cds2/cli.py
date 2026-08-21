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

    print("cognitive-discovery-system-v2")
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
    if len(widths) != 1 or len(a_values) != next(iter(widths)):
        print("error: --a must be square, rows separated by ';'", file=sys.stderr)
        return 1
    b_values = _parse_numbers(args.b)
    solution = solve(np.array(a_values, dtype=float), np.array(b_values, dtype=float))
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cds2", description="cognitive-discovery-system-v2 command line"
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
