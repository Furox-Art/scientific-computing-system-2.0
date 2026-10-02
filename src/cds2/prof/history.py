"""Append-only benchmark history backed by JSONL files.

Each benchmark run writes one JSON line to ``benchmarks/history/<run_id>.jsonl``
inside a source checkout, or to a per-user data directory when cds2 is
installed as a package. ``BenchHistory`` loads a range of runs into a
``pandas.DataFrame`` for analysis and regression gating.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _user_history_dir_for(os_name: str, plat: str, environ: Mapping[str, str], home: Path) -> Path:
    """Pure user-data-directory selection (no environment access).

    ``LOCALAPPDATA`` / ``APPDATA`` on Windows, ``~/Library/Application
    Support`` on macOS and ``XDG_DATA_HOME`` (or ``~/.local/share``)
    elsewhere. Kept parameterised so every platform branch is unit
    testable on any host.
    """
    app = "cds2"
    if os_name == "nt":
        base = environ.get("LOCALAPPDATA") or environ.get("APPDATA") or str(home)
        return Path(base) / app / "history"
    if plat == "darwin":
        return home / "Library" / "Application Support" / app / "history"
    base = environ.get("XDG_DATA_HOME") or str(home / ".local" / "share")
    return Path(base) / app / "history"


def _default_user_history_dir() -> Path:
    """Per-user history directory for installed (non-checkout) layouts.

    Never points inside the Python environment.
    """
    return _user_history_dir_for(os.name, sys.platform, os.environ, Path.home())


def _repo_root() -> Path:
    """Walk up until we find the directory containing ``src/cds2``.

    Falls back to the per-user data directory when cds2 runs from an
    installed package (no checkout layout is visible). The fallback never
    resolves inside the Python environment itself.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "src" / "cds2").is_dir():
            return parent
    return _default_user_history_dir()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


@dataclasses.dataclass(frozen=True)
class BenchEntry:
    """One benchmark result within a run."""

    name: str
    baseline_library: str
    baseline_seconds: float
    cds2_seconds: float

    @property
    def ratio(self) -> float:
        return self.cds2_seconds / self.baseline_seconds if self.baseline_seconds else float("inf")


class BenchHistory:
    """Append-only JSONL history of benchmark runs.

    Constructing an instance never touches the filesystem; the history
    directory is created on first :meth:`append`.

    Args:
        repo_root: path to the repository root. Defaults to the directory
            containing ``src/cds2``, or the per-user data directory when
            cds2 is installed (never inside the Python environment).
        history_dir: path to the history directory. Defaults to
            ``<repo_root>/benchmarks/history`` inside a checkout, or the
            per-user data directory for installed packages.
    """

    def __init__(
        self,
        repo_root: str | Path | None = None,
        history_dir: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else _repo_root()
        if history_dir is not None:
            self.history_dir = Path(history_dir)
        elif (self.repo_root / "src" / "cds2").is_dir():
            self.history_dir = self.repo_root / "benchmarks" / "history"
        else:
            self.history_dir = self.repo_root

    def _run_path(self, run_id: str) -> Path:
        safe = "".join(c for c in run_id if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError(f"invalid run_id: {run_id!r}")
        return self.history_dir / f"{safe}.jsonl"

    def append(
        self,
        entries: Iterable[BenchEntry | dict[str, Any]],
        run_id: str | None = None,
    ) -> Path:
        """Append a run to the history.

        Creates the history directory on first use. Returns the path
        written to. Each entry is serialized as a JSON object with ``name``,
        ``baseline_library``, ``baseline_seconds``, ``cds2_seconds`` and
        ``ratio``.
        """
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        path = self._run_path(run_id)
        env = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "commit": _git_commit(),
            "env": env,
            "entries": [
                dataclasses.asdict(e) if isinstance(e, BenchEntry) else dict(e) for e in entries
            ],
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return path

    def load_range(
        self,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """Load all runs within ``[start, end]`` into a flat DataFrame.

        Each row is one benchmark entry with the run's ``timestamp``,
        ``commit`` and ``env`` joined in. ``start``/``end`` are ISO date
        strings (``YYYY-MM-DD``) or ``date`` objects; ``None`` means unbounded.
        """
        if isinstance(start, date):
            start = start.isoformat()
        if isinstance(end, date):
            end = end.isoformat()

        rows: list[dict[str, Any]] = []
        for path in sorted(self.history_dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = record.get("timestamp", "")
                    if start and ts < start:
                        continue
                    if end and ts > end:
                        continue
                    for entry in record.get("entries", []):
                        rows.append(
                            {
                                "timestamp": ts,
                                "commit": record.get("commit", ""),
                                "env": record.get("env", {}),
                                **entry,
                            }
                        )
        if not rows:
            empty: pd.DataFrame = pd.DataFrame(
                columns=[
                    "timestamp",
                    "commit",
                    "env",
                    "name",
                    "baseline_library",
                    "baseline_seconds",
                    "cds2_seconds",
                    "ratio",
                ]
            )
            return empty
        df: pd.DataFrame = pd.DataFrame(rows)
        if "ratio" not in df.columns:
            df["ratio"] = df["cds2_seconds"] / df["baseline_seconds"]
        return df
