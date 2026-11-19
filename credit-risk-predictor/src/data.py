"""Data loading, cleaning and feature engineering.

The same functions are used for training and for scoring (``src/predict.py``), so a
borrower is always transformed in exactly the same way.
"""

import numpy as np
import pandas as pd

TARGET = "SeriousDlqin2yrs"
SEED = 42

RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
ENGINEERED = [
    "late_code_flag",
    "income_missing",
    "total_past_due",
    "log_income",
    "log_debt_ratio",
    "income_per_dependent",
    "over_limit",
]
# Column order the model was trained on (10 raw + 7 engineered = 17 features).
FEATURE_COLUMNS = RAW_FEATURES + ENGINEERED

# Upper caps used to limit the influence of extreme outliers.
CLIP_UPPER = {
    "RevolvingUtilizationOfUnsecuredLines": 2.0,
    "DebtRatio": 5.0,
    "MonthlyIncome": 50_000,
    "NumberOfOpenCreditLinesAndLoans": 40,
    "NumberRealEstateLoansOrLines": 20,
    "NumberOfDependents": 10,
}
LATE_CAP = 20


def make_synthetic(n=30000, seed=SEED):
    """Synthetic data with the same schema as cs-training.csv (smoke tests only)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 90, n)
    util = rng.beta(0.5, 1.5, n)
    hi = rng.random(n) < 0.03
    util[hi] = rng.uniform(1, 5, hi.sum())
    debt = rng.lognormal(-1, 1, n)
    income = rng.lognormal(8.6, 0.6, n)
    open_lines = rng.poisson(8, n)
    late30 = rng.poisson(0.3, n)
    late60 = rng.poisson(0.1, n)
    late90 = rng.poisson(0.1, n)
    real_estate = rng.poisson(1, n)
    deps = rng.poisson(0.8, n).astype(float)

    logit = (
        -3.9
        + 2.0 * np.clip(util, 0, 1.5)
        + 0.8 * late90
        + 0.4 * late30
        + 0.5 * late60
        - 0.02 * (age - 50)
        + 0.15 * np.log1p(debt)
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    income[rng.random(n) < 0.20] = np.nan
    deps[rng.random(n) < 0.026] = np.nan
    code_rows = rng.random(n) < 0.002
    late30[code_rows] = 98
    late60[code_rows] = 98
    late90[code_rows] = 98

    return pd.DataFrame(
        {
            "Unnamed: 0": np.arange(1, n + 1),
            TARGET: y,
            "RevolvingUtilizationOfUnsecuredLines": util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late30,
            "DebtRatio": debt,
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberOfTimes90DaysLate": late90,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": late60,
            "NumberOfDependents": deps,
        }
    )


def load_data(path, synthetic=False):
    if synthetic:
        print("[data] Using SYNTHETIC data (smoke test only, results are meaningless)")
        return make_synthetic()
    df = pd.read_csv(path)
    print(f"[data] Loaded {path}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def clean_features(df):
    """Column-level cleaning that never drops rows (safe to use at scoring time).

    - 96/98 in the past-due columns are placeholder codes: flag them, then set to NaN
    - cap extreme outliers
    """
    df = df.copy()
    df["late_code_flag"] = (df[LATE_COLS] >= 96).any(axis=1).astype(int)
    for c in LATE_COLS:
        df.loc[df[c] >= 96, c] = np.nan
    for col, cap in CLIP_UPPER.items():
        df[col] = df[col].clip(upper=cap)
    for c in LATE_COLS:
        df[c] = df[c].clip(upper=LATE_CAP)
    return df


def clean(df):
    """Training-time cleaning: drop the index column, invalid ages and duplicates."""
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    n0 = len(df)
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]
    df = clean_features(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[clean] {n0:,} -> {len(df):,} rows after removing invalid ages / duplicates")
    return df


def add_features(X):
    """Add engineered features to a frame that already went through ``clean_features``."""
    X = X.copy()
    X["income_missing"] = X["MonthlyIncome"].isna().astype(int)
    X["total_past_due"] = X[LATE_COLS].sum(axis=1, min_count=1)
    X["log_income"] = np.log1p(X["MonthlyIncome"])
    X["log_debt_ratio"] = np.log1p(X["DebtRatio"])
    X["income_per_dependent"] = X["MonthlyIncome"] / (X["NumberOfDependents"] + 1)
    X["over_limit"] = (X["RevolvingUtilizationOfUnsecuredLines"] > 1).astype(int)
    return X


def engineer(df):
    """Split a cleaned training frame into the model matrix X and the target y."""
    X = add_features(df.drop(columns=[TARGET]))[FEATURE_COLUMNS]
    return X, df[TARGET].astype(int)


def prepare_for_scoring(df):
    """Raw borrower columns -> the 17-feature matrix the model expects (no rows dropped)."""
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return add_features(clean_features(df[RAW_FEATURES]))[FEATURE_COLUMNS]
