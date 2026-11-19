"""Model definitions, hyper-parameter search spaces and threshold selection."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import SEED

XGB_PARAM_DIST = {
    "n_estimators": [200, 300, 400, 600],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 10, 20],
    "gamma": [0, 0.5, 1, 2],
    "reg_lambda": [1, 5, 10],
}


def tune_logistic_regression(X_tr, y_tr, cv):
    """Baseline: median impute -> standardise -> class-weighted logistic regression."""
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    search = GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1, 10]}, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_tr, y_tr)
    return search


def tune_xgboost(X_tr, y_tr, cv, n_iter=25):
    """Main model: XGBoost tuned with randomised search (ROC-AUC, stratified CV)."""
    from xgboost import XGBClassifier  # imported here so other modules work without xgboost

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        importance_type="gain",
        n_jobs=-1,
        random_state=SEED,
    )
    search = RandomizedSearchCV(
        xgb, XGB_PARAM_DIST, n_iter=n_iter, scoring="roc_auc", cv=cv,
        n_jobs=1, random_state=SEED, verbose=1,
    )
    search.fit(X_tr, y_tr)
    return search


def best_f1_threshold(y, p):
    """Probability threshold that maximises F1 (used on out-of-fold train predictions)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[np.argmax(f1)])
