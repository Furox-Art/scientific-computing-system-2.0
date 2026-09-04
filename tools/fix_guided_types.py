"""One-shot fixer for PR #8; removed after the generated commit is verified."""

from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match, found {count}: {old[:80]!r}")
    return text.replace(old, new, 1)


def fix_guided_fit() -> None:
    path = Path("src/cds2/guided_fit.py")
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "from dataclasses import asdict, dataclass\nfrom pathlib import Path\nfrom typing import Literal\n",
        "from collections.abc import Callable, Sequence\nfrom dataclasses import asdict, dataclass\nfrom pathlib import Path\nfrom typing import Literal, cast\n",
    )
    text = replace_once(
        text,
        "import pandas as pd\nimport scipy\n",
        "import pandas as pd\nimport scipy\nfrom numpy.typing import NDArray\n",
    )
    text = replace_once(
        text, "from .optimize import curve_fit\n", "from .optimize import FitResult, curve_fit\n"
    )
    text = replace_once(
        text,
        'TrustLabel = Literal["reliable", "caution", "unreliable"]\n\nMODEL_NAMES',
        'TrustLabel = Literal["reliable", "caution", "unreliable"]\nFloatArray = NDArray[np.float64]\nIndexArray = NDArray[np.intp]\nModelFunc = Callable[..., FloatArray]\nBounds = tuple[Sequence[float] | float, Sequence[float] | float]\n\nMODEL_NAMES',
    )
    text = replace_once(
        text,
        "    x: np.ndarray\n    y: np.ndarray\n    sigma: np.ndarray | None = None",
        "    x: FloatArray\n    y: FloatArray\n    sigma: FloatArray | None = None",
    )
    text = replace_once(
        text,
        "    params: np.ndarray\n    parameter_std: np.ndarray\n    confidence_95: np.ndarray",
        "    params: FloatArray\n    parameter_std: FloatArray\n    confidence_95: FloatArray",
    )
    text = replace_once(text, "    outlier_indices: np.ndarray", "    outlier_indices: IndexArray")

    text = replace_once(
        text,
        "def _linear(x: np.ndarray, a: float, b: float) -> np.ndarray:\n    return a * x + b",
        "def _linear(x: FloatArray, a: float, b: float) -> FloatArray:\n    return np.asarray(a * x + b, dtype=np.float64)",
    )
    text = replace_once(
        text,
        "def _quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:\n    return a * x * x + b * x + c",
        "def _quadratic(x: FloatArray, a: float, b: float, c: float) -> FloatArray:\n    return np.asarray(a * x * x + b * x + c, dtype=np.float64)",
    )
    text = replace_once(
        text,
        "def _exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:\n    return a * np.exp(np.clip(b * x, -700.0, 700.0)) + c",
        "def _exponential(x: FloatArray, a: float, b: float, c: float) -> FloatArray:\n    values = a * np.exp(np.clip(b * x, -700.0, 700.0)) + c\n    return np.asarray(values, dtype=np.float64)",
    )
    text = replace_once(
        text,
        "def _power(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:\n    return a * np.power(np.maximum(x, np.finfo(float).tiny), b) + c",
        "def _power(x: FloatArray, a: float, b: float, c: float) -> FloatArray:\n    values = a * np.power(np.maximum(x, np.finfo(float).tiny), b) + c\n    return np.asarray(values, dtype=np.float64)",
    )
    text = replace_once(
        text,
        "def _logistic(x: np.ndarray, low: float, high: float, k: float, x0: float) -> np.ndarray:\n    return low + (high - low) / (\n        1.0 + np.exp(np.clip(-k * (x - x0), -700.0, 700.0))\n    )",
        "def _logistic(x: FloatArray, low: float, high: float, k: float, x0: float) -> FloatArray:\n    values = low + (high - low) / (\n        1.0 + np.exp(np.clip(-k * (x - x0), -700.0, 700.0))\n    )\n    return np.asarray(values, dtype=np.float64)",
    )
    text = replace_once(text, "_MODEL_FUNCS = {", "_MODEL_FUNCS: dict[ModelName, ModelFunc] = {")

    text = replace_once(
        text,
        "    sigma = None if sigma_column is None else frame[sigma_column].to_numpy(dtype=float)\n    return FitDataset(\n        name=Path(path).stem,\n        x=frame[x_column].to_numpy(dtype=float),\n        y=frame[y_column].to_numpy(dtype=float),",
        "    sigma = (\n        None\n        if sigma_column is None\n        else np.asarray(frame[sigma_column].to_numpy(dtype=float), dtype=np.float64)\n    )\n    return FitDataset(\n        name=Path(path).stem,\n        x=np.asarray(frame[x_column].to_numpy(dtype=float), dtype=np.float64),\n        y=np.asarray(frame[y_column].to_numpy(dtype=float), dtype=np.float64),",
    )
    text = replace_once(
        text,
        "    x = np.asarray(dataset.x, dtype=float).copy()\n    y = np.asarray(dataset.y, dtype=float).copy()\n    sigma = None if dataset.sigma is None else np.asarray(dataset.sigma, dtype=float).copy()",
        "    x: FloatArray = np.asarray(dataset.x, dtype=np.float64).copy()\n    y: FloatArray = np.asarray(dataset.y, dtype=np.float64).copy()\n    sigma: FloatArray | None = (\n        None\n        if dataset.sigma is None\n        else np.asarray(dataset.sigma, dtype=np.float64).copy()\n    )",
    )
    text = replace_once(
        text,
        "def _initial_guess(\n    model: ModelName, x: np.ndarray, y: np.ndarray\n) -> tuple[np.ndarray, tuple[object, object]]:",
        "def _initial_guess(\n    model: ModelName, x: FloatArray, y: FloatArray\n) -> tuple[FloatArray, Bounds]:",
    )
    text = replace_once(
        text,
        "def _fit_arrays(\n    model: ModelName, x: np.ndarray, y: np.ndarray, sigma: np.ndarray | None = None\n):",
        "def _fit_arrays(\n    model: ModelName, x: FloatArray, y: FloatArray, sigma: FloatArray | None = None\n) -> FitResult:",
    )
    text = replace_once(
        text,
        "        _MODEL_FUNCS[model],\n        x,\n        y,\n        p0=p0,\n        sigma=sigma,",
        "        cast(Callable[..., object], _MODEL_FUNCS[model]),\n        x.tolist(),\n        y.tolist(),\n        p0=p0.tolist(),\n        sigma=None if sigma is None else sigma.tolist(),",
    )
    text = replace_once(
        text,
        "def _outliers(residuals: np.ndarray) -> np.ndarray:",
        "def _outliers(residuals: FloatArray) -> IndexArray:",
    )
    text = text.replace("np.zeros(0, dtype=int)", "np.zeros(0, dtype=np.intp)")
    text = replace_once(
        text,
        "def _cross_check(\n    model: ModelName, x: np.ndarray, y: np.ndarray, params: np.ndarray\n) -> float:",
        "def _cross_check(\n    model: ModelName, x: FloatArray, y: FloatArray, params: FloatArray\n) -> float:",
    )
    text = text.replace("dtype=float)", "dtype=np.float64)")
    path.write_text(text, encoding="utf-8")


def fix_cli() -> None:
    path = Path("src/cds2/cli.py")
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from pathlib import Path\n",
        "from pathlib import Path\nfrom typing import Literal, cast\n",
    )
    text = replace_once(
        text,
        'def _guided_report_choice(value: str) -> str:\n    if value != "ask":\n        return value\n    answer = input("Report format (pdf/html/markdown/none) [pdf]: ").strip().lower() or "pdf"\n    if answer not in {"pdf", "html", "markdown", "none"}:\n        raise ValueError("report format must be pdf, html, markdown or none")\n    return answer',
        'ReportChoice = Literal["pdf", "html", "markdown", "none"]\n\n\ndef _guided_report_choice(value: str) -> ReportChoice:\n    answer = (\n        input("Report format (pdf/html/markdown/none) [pdf]: ").strip().lower() or "pdf"\n        if value == "ask"\n        else value\n    )\n    if answer not in {"pdf", "html", "markdown", "none"}:\n        raise ValueError("report format must be pdf, html, markdown or none")\n    return cast(ReportChoice, answer)',
    )
    text = replace_once(
        text,
        'missing_total = sum(int(inspect_dataset(ds)["missing"]) for ds in datasets)',
        'missing_total = sum(cast(int, inspect_dataset(ds)["missing"]) for ds in datasets)',
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    fix_guided_fit()
    fix_cli()
