"""Final one-shot wrapper for PR #8 strict typing fixes."""

from pathlib import Path

from fix_guided_types import fix_cli, fix_guided_fit, replace_once

fix_guided_fit()
fix_cli()

path = Path("src/cds2/guided_fit.py")
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "        cv = _cv_rmse(used, model, seed)\n        std = np.asarray(fit.parameter_std, dtype=np.float64)",
    "        cv = _cv_rmse(used, model, seed)\n        rmse = cast(float, fit.rmse)\n        std = np.asarray(fit.parameter_std, dtype=np.float64)",
)
text = replace_once(text, "                float(fit.rmse),", "                float(rmse),")
path.write_text(text, encoding="utf-8")
