# models/

Saved models and evaluation outputs from the committed run.

| Item | Content |
|---|---|
| `xgb_credit_risk.joblib` | Tuned XGBoost model (loaded by `src/predict.py`) |
| `logreg_baseline.joblib` | Tuned Logistic Regression baseline pipeline (impute, scale, classifier) |
| `metrics.json` | Test-set metrics, confidence intervals, threshold, confusion matrix, best hyper-parameters. The app's *Model performance* tab shows this file |
| `figures/` | EDA overview, ROC, precision-recall, confusion matrix and feature importance plots. The app shows every PNG here |

## Reproduce

```bash
pip install -r requirements.txt
python -m src.train --data data/cs-training.csv
```

- Random seed: 42 (split, cross-validation, search and bootstrap).
- Defaults: 25 randomised-search iterations x 5-fold CV for XGBoost; 4-value grid x 5-fold CV for Logistic Regression. Use `--n-iter 40` for more tuning.
- Run time: depends on your CPU. The XGBoost search alone is 25 x 5 = 125 model fits at the default settings, and the script prints the total time when it finishes. Pass `--synthetic --n-iter 2` for a quick smoke test.
- Writes `metrics.json`, `figures/` and both `.joblib` files here, and `dashboard/powerbi_data/scored_test.csv`.

## Compatibility note

`.joblib` files are pickles, so they must be loaded with a similar scikit-learn and XGBoost version to the one that created them. If loading fails after upgrading libraries, re-run the training command above.
