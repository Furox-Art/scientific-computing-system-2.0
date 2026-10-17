"""Repository-wide test integrity gates.

A passing run must execute every collected test: optional native and data
backends are part of CI provisioning, not exemptions from the test suite.
"""

from __future__ import annotations

import pytest


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    skipped = reporter.stats.get("skipped", ())
    if skipped:
        reporter.write_sep("!", f"QUALITY GATE FAILED: {len(skipped)} skipped test(s)")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
