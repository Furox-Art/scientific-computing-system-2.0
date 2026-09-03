"""cds2.bench — benchmark history persistence and regression reporting."""

from __future__ import annotations

from .regression import RegressionReport, run_regression_check

__all__ = ["RegressionReport", "run_regression_check"]
