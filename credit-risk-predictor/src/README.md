# src/

Reusable code for the whole pipeline. Run everything from the repository root so `import src...` works.

| File | Purpose |
|---|---|
| `data.py` | Load and clean the data, build the 17 model features, synthetic data for smoke tests. `prepare_for_scoring` applies the same transformations to new borrowers without dropping rows |
| `models.py` | Logistic Regression baseline and XGBoost definitions, search spaces, F1-optimal threshold |
| `evaluate.py` | Bootstrap confidence intervals, risk-band assignment, plots |
| `train.py` | Training entry point: `python -m src.train --data data/cs-training.csv`. Saves models, `metrics.json`, figures and `scored_test.csv` |
| `export_scores.py` | Rebuilds `dashboard/powerbi_data/scored_test.csv` from the saved model without retraining |
| `predict.py` | `score(df)` used by `app.py` |

## Interface used by `app.py`

```python
def score(df: pd.DataFrame) -> pd.DataFrame:
    """df has the 10 raw columns of cs-training.csv (no target).
    Returns df with three added columns:
        default_probability   float between 0 and 1
        risk_band             Low / Medium / High
        flagged_high_risk     1 if the probability is at or above the decision threshold, else 0
    """
```

## Risk bands

The decision threshold comes from `models/metrics.json` (F1-optimal on out-of-fold training predictions, 0.210 for the committed model).

| Band | Probability |
|---|---|
| Low | below threshold / 2 (about 10.5%) |
| Medium | from threshold / 2 up to the threshold |
| High | at or above the threshold (`flagged_high_risk = 1`) |
