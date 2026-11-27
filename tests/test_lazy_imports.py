"""Export-surface guards: deduplicated __all__ and lazy deprecated imports.

``cds2.special`` and ``cds2.distributions`` are deprecated since 4.3.0. They
used to be imported eagerly by ``cds2/__init__.py``, so a bare ``import cds2``
emitted both DeprecationWarnings even for users who never touch them. They
are now resolved lazily through module ``__getattr__`` (PEP 562): the API is
unchanged, but the warning fires only on first actual use. Fresh subprocesses
pin both halves of that contract because warning state is process-global.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import cds2

ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str, *flags: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "CDS_PURE": "1",
        "MPLBACKEND": "Agg",
    }
    return subprocess.run(
        [sys.executable, *flags, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


class TestExportSurface:
    def test_all_has_no_duplicates(self) -> None:
        # 34 module names used to be listed twice (535 entries for 501 names).
        assert len(cds2.__all__) == len(set(cds2.__all__))

    def test_every_exported_name_resolves(self) -> None:
        for name in cds2.__all__:
            assert getattr(cds2, name) is not None, name

    def test_new_public_apis_are_exported(self) -> None:
        for name in (
            "split_rhat",
            "effective_sample_size",
            "bonferroni_correction",
            "holm_correction",
            "benjamini_hochberg_correction",
            "ModelComparison",
            "compare_models",
        ):
            assert name in cds2.__all__
            assert getattr(cds2, name) is not None

    def test_lazy_alias_sets_match_the_deprecated_modules(self) -> None:
        assert cds2._SPECIAL_ALIASES == frozenset(cds2.special.__all__)
        assert cds2._DISTRIBUTION_ALIASES == frozenset(cds2.distributions.__all__)
        assert cds2._SPECIAL_ALIASES <= set(cds2.__all__)
        assert cds2._DISTRIBUTION_ALIASES <= set(cds2.__all__)

    def test_dir_includes_lazy_names(self) -> None:
        listing = dir(cds2)
        for name in ("special", "distributions", "erf", "gamma_pdf", "split_rhat"):
            assert name in listing

    def test_unknown_attribute_raises(self) -> None:
        with pytest.raises(AttributeError, match="no attribute 'definitely_missing'"):
            _ = cds2.definitely_missing


class TestLazyDeprecatedImports:
    def test_bare_import_emits_no_deprecation_warning(self) -> None:
        proc = _run_python("import cds2", "-W", "error::DeprecationWarning")
        assert proc.returncode == 0, proc.stderr

    def test_bare_import_does_not_load_the_deprecated_modules(self) -> None:
        proc = _run_python(
            "import sys; import cds2; "
            "assert 'cds2.special' not in sys.modules; "
            "assert 'cds2.distributions' not in sys.modules"
        )
        assert proc.returncode == 0, proc.stderr

    @pytest.mark.parametrize(
        "snippet",
        [
            "cds2.special",
            "cds2.distributions",
            "cds2.erf(0.5)",
            "cds2.gamma_pdf(1.0)",
        ],
    )
    def test_first_use_warns(self, snippet: str) -> None:
        proc = _run_python(f"import cds2; {snippet}", "-W", "error::DeprecationWarning")
        assert proc.returncode != 0
        assert "DeprecationWarning" in proc.stderr
        assert "deprecated since 4.3.0" in proc.stderr

    def test_star_import_still_provides_the_full_surface(self) -> None:
        proc = _run_python(
            "from cds2 import *; "
            "print(float(erf(0.0)[0]), float(gamma_fn(5.0)[0]), "
            "float(student_t_cdf(0.0, 1.0)[0]), special.__name__, distributions.__name__)"
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.split() == ["0.0", "24.0", "0.5", "cds2.special", "cds2.distributions"]

    def test_aliases_still_compute_correct_values(self) -> None:
        assert float(cds2.erf(0.0)[0]) == 0.0
        assert float(cds2.gamma_fn(5.0)[0]) == pytest.approx(24.0)
        assert float(cds2.student_t_cdf(0.0, 1.0)[0]) == pytest.approx(0.5)
        assert float(cds2.norm_pdf(0.0)) == pytest.approx(0.3989422804014327)

    def test_module_access_is_cached(self) -> None:
        assert cds2.special is cds2.special
        assert cds2.distributions is cds2.distributions

    def test_direct_submodule_import_still_works(self) -> None:
        proc = _run_python(
            "import cds2.special as s; import cds2.distributions as d; "
            "print(float(s.erf(0.0)[0]), float(d.uniform_pdf(0.5)[0]))"
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.split() == ["0.0", "1.0"]
