"""Rebuild dashboard/powerbi_data/scored_test.csv from the SAVED model (no retraining).

The 80/20 split is deterministic (seed 42), so this scores exactly the same held-out
test rows that ``src.train`` evaluated.

    python -m src.export_scores --data data/cs-training.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import SEED, clean, engineer, load_data
from .evaluate import assign_risk_band

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="Export the scored held-out test set")
    ap.add_argument("--data", default=str(ROOT / "data" / "cs-training.csv"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "powerbi_data" / "scored_test.csv"))
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    model = joblib.load(models_dir / "xgb_credit_risk.joblib")
    thr = float(json.loads((models_dir / "metrics.json").read_text())["threshold"])

    X, y = engineer(clean(load_data(args.data)))
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

    prob = model.predict_proba(X_te)[:, 1]
    out = pd.DataFrame(
        {
            "row_id": X_te.index,
            "age": X_te["age"].values,
            "RevolvingUtilizationOfUnsecuredLines": X_te["RevolvingUtilizationOfUnsecuredLines"].values,
            "actual_default": y_te.values,
            "default_probability": prob,
            "risk_band": assign_risk_band(prob, thr),
            "flagged_high_risk": (prob >= thr).astype(int),
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out):,} scored rows to {out_path}")


if __name__ == "__main__":
    main()
