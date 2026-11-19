"""Evaluation metrics, bootstrap confidence intervals and plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from .data import SEED, TARGET


def bootstrap_ci(y, p_xgb, p_lr, n_boot=300, seed=SEED):
    """Paired bootstrap 95% CIs for both ROC-AUCs and the relative AUC lift (%)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    a_x, a_l, lift = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        sx = roc_auc_score(y[idx], p_xgb[idx])
        sl = roc_auc_score(y[idx], p_lr[idx])
        a_x.append(sx)
        a_l.append(sl)
        lift.append((sx - sl) / sl * 100)

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    return ci(a_x), ci(a_l), ci(lift)


def assign_risk_band(prob, threshold):
    """Low < threshold/2 <= Medium < threshold <= High. 'High' equals flagged_high_risk."""
    prob = np.asarray(prob, dtype=float)
    return np.where(prob >= threshold, "High", np.where(prob >= threshold / 2, "Medium", "Low"))


def plot_eda(df_raw, df_clean, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    counts = df_clean[TARGET].value_counts().sort_index()
    ax[0].bar(["No default", "Default"], counts.values, color=["#4C78A8", "#E45756"])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax[0].set_title("Class balance")
    miss = df_raw.drop(columns=[c for c in df_raw.columns if c.startswith("Unnamed")]).isna().mean() * 100
    miss = miss[miss > 0].sort_values()
    ax[1].barh(miss.index, miss.values, color="#F58518")
    ax[1].set_title("Missing values in raw data (%)")
    ax[2].hist(df_clean["age"], bins=30, color="#54A24B")
    ax[2].set_title("Age distribution (cleaned)")
    plt.tight_layout()
    plt.savefig(out / "eda_overview.png", dpi=150)
    plt.close()


def plot_roc(y, p_xgb, p_lr, auc_x, auc_l, out):
    plt.figure(figsize=(6, 5))
    for p, name, auc in [(p_xgb, "XGBoost", auc_x), (p_lr, "Logistic Regression", auc_l)]:
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (held-out test set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=150)
    plt.close()


def plot_pr(y, p_xgb, p_lr, ap_x, ap_l, out):
    plt.figure(figsize=(6, 5))
    for p, name, ap in [(p_xgb, "XGBoost", ap_x), (p_lr, "Logistic Regression", ap_l)]:
        prec, rec, _ = precision_recall_curve(y, p)
        plt.plot(rec, prec, label=f"{name} (AP = {ap:.3f})")
    plt.axhline(np.mean(y), color="k", ls="--", alpha=0.5, label="Base rate")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve (held-out test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pr_curve.png", dpi=150)
    plt.close()


def plot_cm(cm, thr, out):
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    plt.xticks([0, 1], ["Pred: No default", "Pred: Default"])
    plt.yticks([0, 1], ["Actual: No default", "Actual: Default"])
    plt.title(f"XGBoost confusion matrix (threshold = {thr:.3f})")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_importance(names, imp, out):
    order = np.argsort(imp)[-12:]
    plt.figure(figsize=(7, 5))
    plt.barh(np.array(names)[order], np.array(imp)[order], color="#4C78A8")
    plt.title("XGBoost feature importance (gain)")
    plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=150)
    plt.close()
