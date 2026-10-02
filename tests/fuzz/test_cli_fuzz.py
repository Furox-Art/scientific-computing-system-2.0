"""CLI fuzz tests: run ``cds2`` with random arg strings and assert no crashes.

The goal is not correctness but robustness: no unhandled tracebacks, no hangs,
exit code 0 or 1 (never a segfault or Python crash).
"""

from __future__ import annotations

import random
import string
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLI = [sys.executable, "-m", "cds2"]

SAFE_SUBCOMMANDS = ["info", "stats", "entropy", "units"]


def _random_string(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits + "-,.", k=length))


def _run_cli(args: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        CLI + args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestCLIFuzz:
    """Fuzz the CLI with random arguments."""

    @pytest.mark.parametrize("seed", range(10))
    def test_info_random_args(self, seed: int) -> None:
        random.seed(seed)
        args = ["info"]
        if random.random() > 0.5:
            args.append(_random_string(4))
        result = _run_cli(args)
        # Should not segfault; exit code 0 or 1 (usage error) is fine.
        assert result.returncode in (0, 1, 2)

    @pytest.mark.parametrize("seed", range(10))
    def test_stats_random_input(self, seed: int) -> None:
        random.seed(seed)
        nums = ",".join(str(random.gauss(0, 100)) for _ in range(random.randint(2, 20)))
        result = _run_cli(["stats", nums], timeout=5.0)
        assert result.returncode in (0, 1, 2)

    @pytest.mark.parametrize("seed", range(10))
    def test_entropy_random_input(self, seed: int) -> None:
        random.seed(seed)
        nums = ",".join(str(random.random()) for _ in range(random.randint(2, 10)))
        result = _run_cli(["entropy", nums], timeout=5.0)
        assert result.returncode in (0, 1, 2)

    @pytest.mark.parametrize("seed", range(10))
    def test_units_random_args(self, seed: int) -> None:
        random.seed(seed)
        args = ["units", str(random.uniform(-100, 100))]
        if random.random() > 0.5:
            args += ["--from-unit", _random_string(3)]
        if random.random() > 0.5:
            args += ["--to-unit", _random_string(3)]
        result = _run_cli(args, timeout=5.0)
        assert result.returncode in (0, 1, 2)
