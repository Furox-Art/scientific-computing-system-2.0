"""Command-line interface: ``cds2``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

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
    if not matrix_rows:
        print("error: --a must contain at least one matrix row", file=sys.stderr)
        return 1
    a_values = [_parse_numbers(row) for row in matrix_rows]
    widths = {len(row) for row in a_values}
    if len(widths) != 1 or len(a_values) != next(iter(widths)):
        print("error: --a must be square, rows separated by ';'", file=sys.stderr)
        return 1
    b_values = _parse_numbers(args.b)
    if len(b_values) != len(a_values):
        print("error: --b length must match the matrix dimension", file=sys.stderr)
        return 1
    try:
        solution = solve(np.array(a_values, dtype=float), np.array(b_values, dtype=float))
    except (ValueError, np.linalg.LinAlgError) as exc:
        print(f"error: could not solve system: {exc}", file=sys.stderr)
        return 1
    print("solution  " + "  ".join(f"{value:.6g}" for value in np.atleast_1d(solution)))
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from .viz import plot_series, save_figure

    numbers = _parse_numbers(args.numbers)
    if not numbers:
        print("error: plot needs at least one number", file=sys.stderr)
        return 1
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
    if not probabilities:
        print("error: entropy needs at least one probability", file=sys.stderr)
        return 1
    if any(value < 0 for value in probabilities):
        print("error: probabilities must be non-negative", file=sys.stderr)
        return 1
    total = sum(probabilities)
    if total <= 0:
        print("error: probabilities must sum to a positive value", file=sys.stderr)
        return 1
    if args.base <= 0 or abs(args.base - 1.0) < 1e-15:
        print("error: --base must be positive and not equal to 1", file=sys.stderr)
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


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    answer = input(prompt + suffix).strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def _guided_report_choice(value: str) -> str:
    if value != "ask":
        return value
    answer = input("Report format (pdf/html/markdown/none) [pdf]: ").strip().lower() or "pdf"
    if answer not in {"pdf", "html", "markdown", "none"}:
        raise ValueError("report format must be pdf, html, markdown or none")
    return answer


def cmd_guided_fit(args: argparse.Namespace) -> int:
    from .guided_fit import (
        MODEL_NAMES,
        inspect_dataset,
        load_csv_dataset,
        plot_result,
        recommend_model,
        run_guided_fit,
        save_manifest,
        write_report,
    )

    try:
        datasets = tuple(load_csv_dataset(path, args.x, args.y, args.sigma) for path in args.files)
        missing_policy = args.missing
        missing_total = sum(int(inspect_dataset(ds)["missing"]) for ds in datasets)
        if missing_policy == "ask":
            if missing_total:
                print(f"missing   {missing_total} values need a decision")
                print("suggest   interpolate missing y values; invalid x/sigma rows are dropped")
                missing_policy = (
                    "interpolate"
                    if _ask_yes_no("Use the suggested missing-data treatment?")
                    else "drop"
                )
            else:
                missing_policy = "interpolate"

        recommendation = recommend_model(datasets, missing_policy=missing_policy, seed=args.seed)
        model = args.model
        if model is None:
            print(f"recommend {recommendation.model}")
            print(f"reason    {recommendation.reason}")
            print(
                f"tradeoff  speed={recommendation.speed}; "
                f"accuracy={recommendation.accuracy}; "
                f"simplicity={recommendation.simplicity}"
            )
            if recommendation.common_model_warning:
                print("warning   one common model is weaker for at least one dataset")
            if _ask_yes_no("Use this model?"):
                model = recommendation.model
            else:
                choice = input(f"Choose model ({'/'.join(MODEL_NAMES)}): ").strip().lower()
                if choice not in MODEL_NAMES:
                    raise ValueError(f"unknown model: {choice}")
                model = choice

        preliminary = run_guided_fit(
            datasets,
            model,
            missing_policy=missing_policy,
            outlier_policy="keep",
            seed=args.seed,
        )
        outlier_total = sum(item.outlier_indices.size for item in preliminary.datasets)
        outlier_policy = args.outliers
        if outlier_policy == "ask":
            if outlier_total:
                print(f"outliers  {outlier_total} suspicious points detected")
                outlier_policy = (
                    "exclude"
                    if _ask_yes_no("Exclude them and refit?", default=False)
                    else "keep"
                )
            else:
                outlier_policy = "keep"
        result = (
            preliminary
            if outlier_policy == "keep"
            else run_guided_fit(
                datasets,
                model,
                missing_policy=missing_policy,
                outlier_policy="exclude",
                seed=args.seed,
            )
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = plot_result(result, datasets, output_dir)
        manifest = save_manifest(
            result,
            datasets,
            output_dir / "guided_fit_manifest.json",
            x_column=args.x,
            y_column=args.y,
            sigma_column=args.sigma,
        )
        report_choice = _guided_report_choice(args.report)
        report_path = None if report_choice == "none" else write_report(result, output_dir, report_choice)

        for item in result.datasets:
            r2 = "undefined" if item.r_squared is None else f"{item.r_squared:.6g}"
            print(
                f"dataset   {item.name}: rmse={item.rmse:.6g}; "
                f"cv_rmse={item.cv_rmse:.6g}; r2={r2}"
            )
        print(f"verdict   {result.trust}: {result.comment}")
        print(f"manifest  {manifest}")
        print(f"plots     {len(plot_paths)} files")
        if report_path is not None:
            print(f"report    {report_path}")

        if result.trust != "reliable":
            alternative = recommend_model(
                datasets,
                missing_policy=missing_policy,
                seed=args.seed,
                exclude=(model,),
            )
            print(f"next      consider {alternative.model}: {alternative.reason}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_guided_fit_rerun(args: argparse.Namespace) -> int:
    from .guided_fit import rerun_manifest

    try:
        result = rerun_manifest(args.manifest)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"model     {result.model}")
    print(f"verdict   {result.trust}: {result.comment}")
    for item in result.datasets:
        print(f"dataset   {item.name}: rmse={item.rmse:.6g}; cv_rmse={item.cv_rmse:.6g}")
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

    guided_parser = subparsers.add_parser(
        "guided-fit",
        help="user-controlled scientific model recommendation, fitting and reporting",
    )
    guided_parser.add_argument("files", nargs="+", help="CSV dataset(s)")
    guided_parser.add_argument("--x", required=True, help="x column")
    guided_parser.add_argument("--y", required=True, help="y column")
    guided_parser.add_argument("--sigma", default=None, help="optional uncertainty column")
    guided_parser.add_argument(
        "--model", choices=["linear", "quadratic", "exponential", "power", "logistic"]
    )
    guided_parser.add_argument(
        "--missing", choices=["ask", "drop", "interpolate"], default="ask"
    )
    guided_parser.add_argument(
        "--outliers", choices=["ask", "keep", "exclude"], default="ask"
    )
    guided_parser.add_argument(
        "--report", choices=["ask", "pdf", "html", "markdown", "none"], default="ask"
    )
    guided_parser.add_argument("--output-dir", default="guided-fit-results")
    guided_parser.add_argument("--seed", type=int, default=0)
    guided_parser.set_defaults(handler=cmd_guided_fit)

    rerun_parser = subparsers.add_parser(
        "guided-fit-rerun", help="repeat a guided fit from a saved manifest"
    )
    rerun_parser.add_argument("manifest")
    rerun_parser.set_defaults(handler=cmd_guided_fit_rerun)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
