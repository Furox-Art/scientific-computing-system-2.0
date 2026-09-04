"""Tests for the cds2 command-line interface."""

import pytest

from cds2.cli import build_parser, main


class TestInfo:
    def test_info_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["info"])
        out = capsys.readouterr().out
        assert code == 0
        assert "cds2" in out
        assert "numpy" in out


class TestStatsCommand:
    def test_descriptive_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["stats", "1,2,3,4,5"])
        out = capsys.readouterr().out
        assert code == 0
        assert "mean" in out
        assert "median" in out

    def test_too_few_numbers_fails(self) -> None:
        assert main(["stats", "42"]) == 1


class TestIntegrateCommand:
    def test_sin_integral(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["integrate", "sin", "--a", "0", "--b", "3.14159265"])
        out = capsys.readouterr().out
        assert code == 0
        assert "value" in out

    def test_unknown_function(self) -> None:
        assert main(["integrate", "tan", "--a", "0", "--b", "1"]) == 1


class TestLinsolveCommand:
    def test_small_system(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["linsolve", "--a", "3,1;1,2", "--b", "9,8"])
        out = capsys.readouterr().out
        assert code == 0
        assert "solution" in out

    def test_non_square_rejected(self) -> None:
        assert main(["linsolve", "--a", "1,2,3;4,5,6", "--b", "1,2"]) == 1

    def test_empty_matrix_rejected_without_traceback(self) -> None:
        assert main(["linsolve", "--a", "", "--b", "1"]) == 1

    def test_rhs_dimension_mismatch_rejected(self) -> None:
        assert main(["linsolve", "--a", "1,0;0,1", "--b", "1"]) == 1

    def test_singular_matrix_rejected_without_traceback(self) -> None:
        assert main(["linsolve", "--a", "1,2;2,4", "--b", "3,6"]) == 1


class TestPlotCommand:
    def test_ascii_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["plot", "1,5,3"])
        out = capsys.readouterr().out
        assert code == 0
        assert "|" in out

    def test_png_output(self, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
        import matplotlib

        matplotlib.use("Agg")
        target = tmp_path / "cli.png"
        code = main(["plot", "1,2,3", "--file", str(target)])
        out = capsys.readouterr().out
        assert code == 0
        assert "saved" in out
        assert target.exists()

    def test_empty_plot_rejected_without_traceback(self) -> None:
        assert main(["plot", ""]) == 1


class TestParser:
    def test_requires_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_unknown_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["frobnicate"])


class TestEntropyCommand:
    def test_uniform_entropy(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["entropy", "0.25,0.25,0.25,0.25"])
        out = capsys.readouterr().out
        assert code == 0
        assert "2.000000" in out

    def test_nats_base(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["entropy", "0.5,0.5", "--base", "2.718281828459045"])
        assert code == 0

    def test_empty_input_fails(self) -> None:
        assert main(["entropy", ""]) == 1

    def test_nonpositive_total_fails(self) -> None:
        assert main(["entropy", "0,0"]) == 1

    def test_negative_probability_fails(self) -> None:
        assert main(["entropy", "1,-0.2,0.2"]) == 1

    def test_invalid_log_base_one_fails(self) -> None:
        assert main(["entropy", "0.5,0.5", "--base", "1"]) == 1

    def test_invalid_log_base_zero_fails(self) -> None:
        assert main(["entropy", "0.5,0.5", "--base", "0"]) == 1


class TestUnitsCommand:
    def test_length_conversion(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["units", "5", "--from-unit", "km", "--to-unit", "mile"])
        out = capsys.readouterr().out
        assert code == 0
        assert "3.106855961" in out

    def test_temperature_conversion(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["units", "25", "--from-unit", "C", "--to-unit", "F"])
        out = capsys.readouterr().out
        assert code == 0
        assert "77" in out

    def test_dimension_mixing_fails(self) -> None:
        assert main(["units", "1", "--from-unit", "m", "--to-unit", "kg"]) == 1

    def test_unknown_unit_fails(self) -> None:
        assert main(["units", "1", "--from-unit", "parsec", "--to-unit", "m"]) == 1


class TestSolveCommand:
    def test_quadratic_roots(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["solve", "--coeffs", "1,-5,6"])
        out = capsys.readouterr().out
        assert code == 0
        assert "real      2" in out
        assert "real      3" in out

    def test_complex_roots(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["solve", "--coeffs", "1,0,1"])
        out = capsys.readouterr().out
        assert code == 0
        assert "complex" in out

    def test_zero_leading_coefficient_fails(self) -> None:
        assert main(["solve", "--coeffs", "0,1,-1"]) == 1

    def test_trailing_zeros_allowed(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["solve", "--coeffs", "1,0,0"])
        out = capsys.readouterr().out
        assert code == 0
        assert "real" in out
