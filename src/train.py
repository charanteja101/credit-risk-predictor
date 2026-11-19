"""Train, evaluate and save the credit risk models.

Run from the repository root:

    python -m src.train --data data/cs-training.csv
    python -m src.train --data data/cs-training.csv --n-iter 40   # more tuning
    python -m src.train --synthetic --n-iter 2                    # smoke test, no download

Outputs
-------
models/xgb_credit_risk.joblib, models/logreg_baseline.joblib, models/metrics.json
models/figures/*.png
dashboard/powerbi_data/scored_test.csv   (held-out test set with model scores; git-ignored)
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .data import SEED, clean, engineer, load_data
from .evaluate import (
    assign_risk_band,
    bootstrap_ci,
    plot_cm,
    plot_eda,
    plot_importance,
    plot_pr,
    plot_roc,
)
from .models import best_f1_threshold, tune_logistic_regression, tune_xgboost

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="Train the Credit Risk Predictor")
    ap.add_argument("--data", default=str(ROOT / "data" / "cs-training.csv"), help="path to cs-training.csv")
    ap.add_argument("--models-dir", default=str(ROOT / "models"), help="where to save models, metrics and figures")
    ap.add_argument(
        "--scores-out",
        default=str(ROOT / "dashboard" / "powerbi_data" / "scored_test.csv"),
        help="where to save the scored test set (used by SQL and Power BI)",
    )
    ap.add_argument("--n-iter", type=int, default=25, help="RandomizedSearchCV iterations")
    ap.add_argument("--cv", type=int, default=5, help="CV folds")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (smoke test)")
    args = ap.parse_args()

    t_start = time.time()
    models_dir = Path(args.models_dir)
    fig_dir = models_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = load_data(args.data, args.synthetic)
    df = clean(raw)
    X, y = engineer(df)
    plot_eda(raw, df, fig_dir)
    print(f"[data] {X.shape[1]} model features | default rate = {y.mean():.2%}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    print(f"[split] train = {len(X_tr):,} | test = {len(X_te):,}")
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED)

    print("\n[baseline] Tuning Logistic Regression ...")
    lr_search = tune_logistic_regression(X_tr, y_tr, cv)
    lr = lr_search.best_estimator_
    print(f"  best C = {lr_search.best_params_['clf__C']} | CV ROC-AUC = {lr_search.best_score_:.4f}")

    print("\n[xgboost] Randomized search ...")
    xgb_search = tune_xgboost(X_tr, y_tr, cv, n_iter=args.n_iter)
    best_xgb = xgb_search.best_estimator_
    print(f"  best CV ROC-AUC = {xgb_search.best_score_:.4f}")
    print(f"  best params     = {xgb_search.best_params_}")

    # Threshold is chosen on out-of-fold TRAIN predictions so the test set is never used for it.
    oof = cross_val_predict(best_xgb, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thr = best_f1_threshold(y_tr, oof)
    print(f"\n[threshold] F1-optimal threshold (from OOF train preds) = {thr:.3f}")

    p_x = best_xgb.predict_proba(X_te)[:, 1]
    p_l = lr.predict_proba(X_te)[:, 1]
    auc_x, auc_l = roc_auc_score(y_te, p_x), roc_auc_score(y_te, p_l)
    ap_x, ap_l = average_precision_score(y_te, p_x), average_precision_score(y_te, p_l)
    lift = (auc_x - auc_l) / auc_l * 100
    pred = (p_x >= thr).astype(int)
    cm = confusion_matrix(y_te, pred)
    ci_x, ci_l, ci_lift = bootstrap_ci(y_te.values, p_x, p_l)

    metrics = {
        "n_rows_clean": int(len(df)),
        "n_features": int(X.shape[1]),
        "default_rate": float(y.mean()),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "xgb_cv_auc": float(xgb_search.best_score_),
        "xgb_test_auc": float(auc_x),
        "xgb_test_auc_95ci": ci_x,
        "lr_cv_auc": float(lr_search.best_score_),
        "lr_test_auc": float(auc_l),
        "lr_test_auc_95ci": ci_l,
        "auc_lift_pct": float(lift),
        "auc_lift_pct_95ci": ci_lift,
        "xgb_test_pr_auc": float(ap_x),
        "lr_test_pr_auc": float(ap_l),
        "threshold": thr,
        "accuracy": float(accuracy_score(y_te, pred)),
        "majority_class_accuracy": float(1 - y_te.mean()),
        "precision": float(precision_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)),
        "confusion_matrix": cm.tolist(),
        "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in xgb_search.best_params_.items()},
        "lr_best_C": float(lr_search.best_params_["clf__C"]),
    }
    imp = np.asarray(best_xgb.feature_importances_)
    metrics["top_features"] = [X.columns[i] for i in np.argsort(imp)[::-1][:5]]

    plot_roc(y_te, p_x, p_l, auc_x, auc_l, fig_dir)
    plot_pr(y_te, p_x, p_l, ap_x, ap_l, fig_dir)
    plot_cm(cm, thr, fig_dir)
    plot_importance(list(X.columns), imp, fig_dir)
    joblib.dump(best_xgb, models_dir / "xgb_credit_risk.joblib")
    joblib.dump(lr, models_dir / "logreg_baseline.joblib")
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Scored held-out test set: input for the SQL lift query and the Power BI dashboard.
    scores_path = Path(args.scores_out)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": X_te.index,
            "age": X_te["age"].values,
            "RevolvingUtilizationOfUnsecuredLines": X_te["RevolvingUtilizationOfUnsecuredLines"].values,
            "actual_default": y_te.values,
            "default_probability": p_x,
            "risk_band": assign_risk_band(p_x, thr),
            "flagged_high_risk": pred,
        }
    ).to_csv(scores_path, index=False)

    print("\n" + "=" * 64)
    print("TEST-SET RESULTS (held-out 20%)")
    print("=" * 64)
    print(f"XGBoost ROC-AUC        : {auc_x:.4f}   95% CI {ci_x[0]:.4f}-{ci_x[1]:.4f}")
    print(f"LogReg  ROC-AUC        : {auc_l:.4f}   95% CI {ci_l[0]:.4f}-{ci_l[1]:.4f}")
    print(f"Relative AUC lift      : {lift:.2f}%   95% CI {ci_lift[0]:.2f}%-{ci_lift[1]:.2f}%")
    print(f"XGBoost PR-AUC         : {ap_x:.4f}  (LogReg {ap_l:.4f}, base rate {y_te.mean():.4f})")
    print(f"\nAt threshold {thr:.3f}:")
    print(f"  Accuracy  {metrics['accuracy']:.4f}  (always-predict-'no default' = {metrics['majority_class_accuracy']:.4f})")
    print(f"  Precision {metrics['precision']:.4f} | Recall {metrics['recall']:.4f} | F1 {metrics['f1']:.4f}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]] = {cm.tolist()}")
    print(f"\nTop features: {', '.join(metrics['top_features'][:3])}")
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Outputs saved to: {models_dir.resolve()}")


if __name__ == "__main__":
    main()
