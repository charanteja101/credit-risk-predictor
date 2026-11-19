"""Scoring used by ``app.py``.

    from src.predict import score
    scored = score(df)   # df has the 10 raw columns of cs-training.csv

Run commands from the repository root so ``import src...`` works.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .data import FEATURE_COLUMNS, prepare_for_scoring
from .evaluate import assign_risk_band

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@lru_cache(maxsize=1)
def _load():
    model_path = MODELS_DIR / "xgb_credit_risk.joblib"
    metrics_path = MODELS_DIR / "metrics.json"
    if not model_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            "Trained model not found in models/. Run `python -m src.train --data data/cs-training.csv` first."
        )
    model = joblib.load(model_path)
    threshold = float(json.loads(metrics_path.read_text())["threshold"])
    return model, threshold


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with default_probability, risk_band and flagged_high_risk added."""
    model, threshold = _load()
    X = prepare_for_scoring(df)
    columns = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    prob = model.predict_proba(X[columns])[:, 1]

    out = df.copy()
    out["default_probability"] = prob
    out["risk_band"] = assign_risk_band(prob, threshold)
    out["flagged_high_risk"] = (prob >= threshold).astype(int)
    return out
