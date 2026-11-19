"""Render the two dashboard pages described in dashboard/README.md as PNG images.

Reads the scored held-out test set and applies the same logic as dashboard/measures.dax
(age and utilisation bands, outcome, precision, recall, lift), so the numbers match
what a Power BI report built from the same file would show.

    python dashboard/build_dashboard.py
    python dashboard/build_dashboard.py --scores path/to/scored_test.csv --out dashboard/screenshots
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BLUE, RED, ORANGE, GREEN, GREY = "#4C78A8", "#E45756", "#F58518", "#54A24B", "#9AA0A6"
BAND_COLORS = {"Low": GREEN, "Medium": ORANGE, "High": RED}
AGE_ORDER = ["Under 25", "25-34", "35-44", "45-54", "55-64", "65+"]
UTIL_ORDER = ["Under 10%", "10-30%", "30-60%", "60-100%", "Over limit"]


def add_columns(df):
    """Calculated columns from measures.dax (labels without the sort-order prefixes)."""
    df = df.copy()
    df["age_band"] = pd.cut(
        df["age"], [-np.inf, 25, 35, 45, 55, 65, np.inf], right=False, labels=AGE_ORDER
    )
    df["utilisation_band"] = pd.cut(
        df["RevolvingUtilizationOfUnsecuredLines"],
        [-np.inf, 0.1, 0.3, 0.6, 1.0, np.inf],
        right=False,
        labels=UTIL_ORDER,
    )
    flag, act = df["flagged_high_risk"] == 1, df["actual_default"] == 1
    df["outcome"] = np.select(
        [flag & act, flag & ~act, ~flag & act],
        ["True positive", "False positive", "False negative"],
        default="True negative",
    )
    return df


def rate_by(df, col, order):
    g = df.groupby(col, observed=False)["actual_default"].agg(["mean", "size"]).reindex(order)
    return g["mean"] * 100, g["size"]


def card(ax, title, value, sub=""):
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="#F4F6F8", ec="#D9DEE3"))
    ax.text(0.5, 0.74, title, ha="center", va="center", fontsize=11, color="#555", transform=ax.transAxes)
    ax.text(0.5, 0.42, value, ha="center", va="center", fontsize=26, fontweight="bold", color="#222", transform=ax.transAxes)
    ax.text(0.5, 0.14, sub, ha="center", va="center", fontsize=9, color="#777", transform=ax.transAxes)


def bar_rates(ax, rates, base, title, color=BLUE, legend_loc="upper right"):
    bars = ax.bar(rates.index.astype(str), rates.values, color=color)
    ax.axhline(base, color="k", ls="--", lw=1, alpha=0.6, label=f"Portfolio average ({base:.1f}%)")
    ax.legend(loc=legend_loc, fontsize=9, frameon=False)
    for b, v in zip(bars, rates.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual default rate (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)


def page1(df, out):
    total, defaults = len(df), int(df["actual_default"].sum())
    base = 100 * defaults / total
    flagged = int(df["flagged_high_risk"].sum())

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.suptitle("Credit Risk Dashboard  |  Page 1: Portfolio risk (held-out test set)", x=0.04, ha="left",
                 fontsize=17, fontweight="bold")
    gs = fig.add_gridspec(2, 6, height_ratios=[0.5, 1.5], hspace=0.38, wspace=0.55,
                          left=0.05, right=0.97, top=0.88, bottom=0.07)
    card(fig.add_subplot(gs[0, 0:2]), "Total borrowers", f"{total:,}")
    card(fig.add_subplot(gs[0, 2:4]), "Actual default rate", f"{base:.2f}%", f"{defaults:,} defaults")
    card(fig.add_subplot(gs[0, 4:6]), "Flagged high risk", f"{flagged:,}", f"{flagged / total:.1%} of borrowers")

    r, _ = rate_by(df, "age_band", AGE_ORDER)
    bar_rates(fig.add_subplot(gs[1, 0:2]), r, base, "Default rate by age band")
    r, _ = rate_by(df, "utilisation_band", UTIL_ORDER)
    bar_rates(fig.add_subplot(gs[1, 2:4]), r, base, "Default rate by utilisation band", color=ORANGE, legend_loc="upper left")

    ax = fig.add_subplot(gs[1, 4:6])
    counts = df["risk_band"].value_counts().reindex(["Low", "Medium", "High"])
    wedges, _ = ax.pie(counts.values, colors=[BAND_COLORS[b] for b in counts.index], startangle=90,
                       counterclock=False, wedgeprops=dict(width=0.38, edgecolor="white"))
    ax.legend(wedges, [f"{b}: {v:,} ({v / total:.1%})" for b, v in counts.items()],
              loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=9, frameon=False)
    ax.set_title("Borrowers by risk band", loc="left", fontsize=12, fontweight="bold")
    fig.savefig(out / "page1_portfolio_risk.png", dpi=110)
    plt.close(fig)


def page2(df, out):
    tp = int((df.outcome == "True positive").sum())
    fp = int((df.outcome == "False positive").sum())
    fn = int((df.outcome == "False negative").sum())
    tn = int((df.outcome == "True negative").sum())
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    base = (tp + fn) / len(df)
    lift = precision / base

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.suptitle("Credit Risk Dashboard  |  Page 2: Model performance (held-out test set)", x=0.04, ha="left",
                 fontsize=17, fontweight="bold")
    gs = fig.add_gridspec(2, 6, height_ratios=[0.5, 1.5], hspace=0.38, wspace=0.9,
                          left=0.05, right=0.97, top=0.88, bottom=0.07)
    card(fig.add_subplot(gs[0, 0:2]), "Precision", f"{precision:.1%}", "flagged borrowers who default")
    card(fig.add_subplot(gs[0, 2:4]), "Recall", f"{recall:.1%}", "defaulters that are flagged")
    card(fig.add_subplot(gs[0, 4:6]), "Lift vs average", f"{lift:.1f}x", "default rate in flagged group / overall")

    ax = fig.add_subplot(gs[1, 0:3])
    g = df.groupby("risk_band")["actual_default"].agg(["mean", "size"]).reindex(["Low", "Medium", "High"])
    bars = ax.bar(g.index, g["mean"] * 100, color=[BAND_COLORS[b] for b in g.index])
    for b, v, n in zip(bars, g["mean"] * 100, g["size"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=10)
    ax.axhline(base * 100, color="k", ls="--", lw=1, alpha=0.6, label=f"Portfolio average ({base:.1%})")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_ylim(0, g["mean"].max() * 100 * 1.25)
    ax.set_title("Actual default rate by risk band (rises from low to high)", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual default rate (%)")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, 3:6])
    labels = ["True negative", "False positive", "False negative", "True positive"]
    vals = [tn, fp, fn, tp]
    bars = ax.barh(labels[::-1], vals[::-1], color=[GREEN, RED, ORANGE, GREY][::-1])
    for b, v in zip(bars, vals[::-1]):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:,}", va="center", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_title("Prediction outcomes at the decision threshold", loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "page2_model_performance.png", dpi=110)
    plt.close(fig)
    return precision, recall, lift


def main():
    ap = argparse.ArgumentParser(description="Render the credit risk dashboard pages as PNG images")
    ap.add_argument("--scores", default=str(ROOT / "dashboard" / "powerbi_data" / "scored_test.csv"))
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "screenshots"))
    args = ap.parse_args()

    scores = Path(args.scores)
    if not scores.exists():
        raise SystemExit(f"{scores} not found. Run `python -m src.export_scores --data data/cs-training.csv` first.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = add_columns(pd.read_csv(scores))
    page1(df, out)
    p, r, lift = page2(df, out)
    print(f"Wrote 2 dashboard images to {out}  (precision {p:.1%}, recall {r:.1%}, lift {lift:.1f}x)")


if __name__ == "__main__":
    main()
