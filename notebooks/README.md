# notebooks/

| Notebook | Content |
|---|---|
| `01_eda.ipynb` | Shape, class balance, missing values, data-quality problems (invalid ages, placeholder codes, outliers, duplicates), default rate by age, utilisation and past-due history |
| `02_experiments.ipynb` | Effect of the engineered features on the baseline, XGBoost on a validation split carved out of the training data, threshold analysis, feature importance |

The notebooks are committed without outputs. Run them from this folder after downloading the data (`data/README.md`):

```bash
cd notebooks
jupyter lab
```

They add the repository root to `sys.path` so they can import from `src/`. The final test set is never used in the experiments; the final pipeline is `python -m src.train`.
