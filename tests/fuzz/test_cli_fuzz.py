"""CLI fuzz tests: run ``cds2`` with random arg strings and assert no crashes.

The goal is not correctness but robustness: no unhandled tracebacks, no hangs,
and no death by signal. Exit status alone cannot carry that, because Python
exits ``1`` both for a handled validation error (``cmd_*`` handlers return 1)
*and* for an uncaught exception. A crash and a usage error are therefore
indistinguishable by return code, so :func:`assert_healthy` additionally
requires that stderr carries no traceback and that the process was not killed by
a signal.

This harness previously ran ``python -m cds2`` while no ``cds2.__main__``
existed, so every invocation died with "No module named cds2.__main__", exited
1, and passed the old ``returncode in (0, 1, 2)`` assertion without ever
executing the CLI. ``cds2/__main__.py`` now exists and
:func:`test_entry_point_actually_runs` guards that this stays true.
"""

from __future__ import annotations

import random
import string
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# ``python -m cds2`` is the documented module entry point. It is used instead of
# ``cds2.cli`` because the latter triggers a runpy RuntimeWarning on every
# invocation (``cds2/__init__.py`` imports ``cli`` eagerly), which would pollute
# the stderr this harness inspects.
CLI = [sys.executable, "-m", "cds2"]

# A cold start imports the whole scientific stack. Measured on the slowest host
# used for this repository: median 2.1 s, worst observed 7.7 s (a 5.0 s timeout
# failed the macOS / Python 3.10 leg). 30 s is roughly 4x the worst legitimate
# cost while still bounding a genuine hang -- verified against a control that
# sleeps for 60 s, which a 30 s timeout catches.
DEFAULT_TIMEOUT = 30.0

TRACEBACK_MARKER = "Traceback (most recent call last)"


def _random_string(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits + "-,.", k=length))


def _run_cli(args: list[str], timeout: float = DEFAULT_TIMEOUT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        CLI + args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def assert_healthy(result: subprocess.CompletedProcess[str], args: list[str]) -> None:
    """Assert the invocation behaved per the CLI's real contract.

    ``cds2.cli.main`` returns ``0`` on success and ``1`` from a handler for a
    handled validation error; ``argparse`` exits ``2`` on a usage error. Those
    three are the only legitimate statuses, and each is checked together with
    the two failure modes a return code cannot express.
    """
    where = f"args={args!r}"

    # A negative return code means the process was killed by a signal, e.g. a
    # segfault in a compiled kernel. No legitimate exit is negative.
    assert result.returncode >= 0, f"{where}: killed by signal {-result.returncode}"

    # Exit 1 covers both a handled error and an uncaught exception, so the
    # traceback check is what actually rules out a crash.
    assert TRACEBACK_MARKER not in result.stderr, (
        f"{where}: unhandled traceback on stderr\n{result.stderr}"
    )

    # The entry point itself must exist. Without this, a missing or renamed
    # ``__main__`` module reproduces the original defect: exit 1, no traceback,
    # and every assertion above still satisfied.
    assert "No module named" not in result.stderr, (
        f"{where}: module entry point missing\n{result.stderr}"
    )
    assert "runpy" not in result.stderr or "RuntimeWarning" not in result.stderr, (
        f"{where}: runpy double-import warning means the wrong module was executed\n{result.stderr}"
    )

    assert result.returncode in (0, 1, 2), f"{where}: unexpected exit {result.returncode}"


def test_entry_point_actually_runs() -> None:
    """Canary: the fuzz cases below are only meaningful if the CLI really runs.

    ``info`` takes no arguments and always succeeds, so it must exit 0 and print
    the distribution banner. This is the assertion that would have caught the
    original defect, where every invocation exited 1 without executing anything.
    """
    result = _run_cli(["info"])
    assert result.returncode == 0, (
        f"python -m cds2 info exited {result.returncode}\n{result.stderr}"
    )
    assert "scientific-computing-system-2.0" in result.stdout, result.stdout
    assert "cds2" in result.stdout, result.stdout
    assert_healthy(result, ["info"])


def test_entry_point_help_exits_zero() -> None:
    """``--help`` is the cheapest end-to-end check that argparse is wired up."""
    result = _run_cli(["--help"])
    assert result.returncode == 0, f"--help exited {result.returncode}\n{result.stderr}"
    assert "usage: cds2" in result.stdout, result.stdout
    assert_healthy(result, ["--help"])


def test_usage_error_exits_two() -> None:
    """An unknown subcommand is a usage error, which argparse reports as exit 2.

    This pins the meaning of each status the fuzz harness tolerates, so a change
    in exit-code conventions is caught here rather than silently absorbed.
    """
    result = _run_cli(["definitely-not-a-command"])
    assert result.returncode == 2, f"expected argparse usage exit 2, got {result.returncode}"
    assert_healthy(result, ["definitely-not-a-command"])


class TestCLIFuzz:
    """Fuzz the CLI with random arguments."""

    @pytest.mark.parametrize("seed", range(10))
    def test_info_random_args(self, seed: int) -> None:
        random.seed(seed)
        args = ["info"]
        if random.random() > 0.5:
            args.append(_random_string(4))
        assert_healthy(_run_cli(args), args)

    @pytest.mark.parametrize("seed", range(10))
    def test_stats_random_input(self, seed: int) -> None:
        random.seed(seed)
        nums = ",".join(str(random.gauss(0, 100)) for _ in range(random.randint(2, 20)))
        args = ["stats", nums]
        assert_healthy(_run_cli(args), args)

    @pytest.mark.parametrize("seed", range(10))
    def test_entropy_random_input(self, seed: int) -> None:
        random.seed(seed)
        nums = ",".join(str(random.random()) for _ in range(random.randint(2, 10)))
        args = ["entropy", nums]
        assert_healthy(_run_cli(args), args)

    @pytest.mark.parametrize("seed", range(10))
    def test_units_random_args(self, seed: int) -> None:
        random.seed(seed)
        args = ["units", str(random.uniform(-100, 100))]
        if random.random() > 0.5:
            args += ["--from-unit", _random_string(3)]
        if random.random() > 0.5:
            args += ["--to-unit", _random_string(3)]
        assert_healthy(_run_cli(args), args)
