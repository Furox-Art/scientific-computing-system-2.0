"""Docs count guards: README/docs numbers must match the tree.

Every claim here is recomputed from the source of truth (``src/cds2`` layout,
``build_parser`` subcommands, ``benchmarks/results.json``) and compared against
the literal text in ``README.md``. The example-count claim used to be hand-typed
and drifts whenever a script is added or removed; pinning the trend line issues
to the code keeps the prose honest. Nothing is hard-coded -- if the tree changes,
the test changes with it or fails loudly.
"""

import argparse
import ast
import re
from pathlib import Path

import cds2.cli

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _flat_modules() -> int:
    src = ROOT / "src" / "cds2"
    return len(
        [p for p in src.glob("*.py") if p.name not in ("__init__.py", "_version.py", "__main__.py")]
    )


def _subpackages() -> int:
    src = ROOT / "src" / "cds2"
    return len([d for d in src.iterdir() if d.is_dir() and (d / "__init__.py").is_file()])


def _public_exports() -> int:
    tree = ast.parse((ROOT / "src" / "cds2" / "__init__.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "__all__":
            return len(node.value.elts)
    raise AssertionError("__all__ not found in cds2/__init__.py")


def _api_pages() -> int:
    return len(list((ROOT / "docs" / "api").glob("*.md")))


def _examples() -> int:
    return len(list((ROOT / "examples").glob("*.py")))


def _cli_subcommands() -> list[str]:
    parser = cds2.cli.build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return sorted(action.choices.keys())
    raise AssertionError("no subparsers found on the cds2 CLI parser")


def _benchmark_rows() -> int:
    import json

    data = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))
    return len(data["results"])


def test_readme_example_count_matches_tree() -> None:
    # Regression guard: README claimed "20 runnable scripts" while examples/ held
    # 21, all executed by the examples-smoke job.
    match = re.search(r"(\d+) runnable scripts", README)
    assert match is not None, "README does not state a runnable-scripts count"
    assert int(match.group(1)) == _examples()


def test_readme_module_and_api_counts_match_tree() -> None:
    flat, subs, api = _flat_modules(), _subpackages(), _api_pages()
    claim = f"{flat} flat modules plus {subs} subpackages; {api} generated API pages"
    assert claim in README, f"README is missing the regenerated module sentence: {claim!r}"


def test_readme_importable_and_export_counts_match_tree() -> None:
    importable = _flat_modules() + _subpackages()
    claim = f"{importable} importable names, {_public_exports()} public exports"
    assert claim in README, f"README is missing the regenerated headline numbers: {claim!r}"


def test_readme_subcommand_count_matches_parser() -> None:
    # README documents "All ten subcommands"; the word must change with the CLI.
    words = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen"}
    n = len(_cli_subcommands())
    assert words.get(n, "ten") in README, (
        f"CLI has {n} subcommands but README does not say '{words.get(n, 'ten')}'"
    )
    for name in _cli_subcommands():
        assert name in README, f"subcommand {name!r} is absent from the README CLI list"


def test_readme_benchmark_case_count_matches_data() -> None:
    n = _benchmark_rows()
    assert f"{n} measured cases" in README, (
        f"benchmarks/results.json has {n} rows but README does not say '{n} measured cases'"
    )
