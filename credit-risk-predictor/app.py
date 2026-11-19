import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "models" / "metrics.json"
FIG_DIR = ROOT / "models" / "figures"

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

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")
st.title("Credit Risk Predictor")
st.caption(
    "Estimates the probability that a borrower becomes seriously delinquent (90+ days past due) within 2 years. "
    "Portfolio project, not a lending decision tool."
)

try:
    from src.predict import score
except ImportError:
    st.error(
        "`src/predict.py` with a `score(df)` function was not found. See `src/README.md` for the interface this app expects."
    )
    st.stop()

tab_single, tab_batch, tab_perf = st.tabs(["Single borrower", "Batch scoring", "Model performance"])

with tab_single:
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age", min_value=18, max_value=100, value=45, step=1)
        income_known = st.checkbox("Monthly income is known", value=True)
        income = st.number_input("Monthly income", min_value=0, max_value=50000, value=5400, step=100)
        dependents = st.number_input("Number of dependents", min_value=0, max_value=10, value=1, step=1)
        open_lines = st.number_input("Open credit lines and loans", min_value=0, max_value=40, value=8, step=1)
        real_estate = st.number_input("Real estate loans / lines", min_value=0, max_value=20, value=1, step=1)
    with right:
        utilisation = st.number_input(
            "Revolving utilisation (1.0 = at credit limit)", min_value=0.0, max_value=2.0, value=0.3, step=0.05
        )
        debt_ratio = st.number_input(
            "Debt ratio (monthly debt payments / income)", min_value=0.0, max_value=5.0, value=0.35, step=0.05
        )
        late_30 = st.number_input("Times 30-59 days past due", min_value=0, max_value=20, value=0, step=1)
        late_60 = st.number_input("Times 60-89 days past due", min_value=0, max_value=20, value=0, step=1)
        late_90 = st.number_input("Times 90+ days late", min_value=0, max_value=20, value=0, step=1)

    row = pd.DataFrame(
        [
            {
                "RevolvingUtilizationOfUnsecuredLines": utilisation,
                "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse": late_30,
                "DebtRatio": debt_ratio,
                "MonthlyIncome": income if income_known else float("nan"),
                "NumberOfOpenCreditLinesAndLoans": open_lines,
                "NumberOfTimes90DaysLate": late_90,
                "NumberRealEstateLoansOrLines": real_estate,
                "NumberOfTime60-89DaysPastDueNotWorse": late_60,
                "NumberOfDependents": dependents,
            }
        ]
    )
    try:
        scored = score(row)
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()
    prob = float(scored["default_probability"].iloc[0])

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Predicted default probability", f"{prob:.1%}")
    m2.metric("Risk band", str(scored["risk_band"].iloc[0]))
    if int(scored["flagged_high_risk"].iloc[0]) == 1:
        st.warning("Flagged high risk: the probability is at or above the decision threshold.")
    else:
        st.success("Not flagged: the probability is below the decision threshold.")

with tab_batch:
    st.write("Upload a CSV with these columns (same format as `cs-training.csv`):")
    st.code(", ".join(RAW_FEATURES))
    upload = st.file_uploader("CSV file", type=["csv"])
    if upload is not None:
        data = pd.read_csv(upload)
        missing = [c for c in RAW_FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            result = score(data)
            st.write(f"Scored {len(result):,} borrowers. Flagged high risk: {int(result['flagged_high_risk'].sum()):,}")
            st.dataframe(result.sort_values("default_probability", ascending=False).head(500))
            st.download_button(
                "Download scored CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="scored_borrowers.csv",
                mime="text/csv",
            )

with tab_perf:
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())

        st.subheader("Held-out test set results")
        st.caption(
            f"{m['n_test']:,} borrowers never used for training or tuning. "
            f"Default rate {m['default_rate']:.1%}, so accuracy alone is misleading."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("XGBoost ROC-AUC", f"{m['xgb_test_auc']:.3f}")
        c1.caption(f"95% CI {m['xgb_test_auc_95ci'][0]:.3f} to {m['xgb_test_auc_95ci'][1]:.3f}")
        c2.metric("Logistic Regression ROC-AUC", f"{m['lr_test_auc']:.3f}")
        c2.caption(f"95% CI {m['lr_test_auc_95ci'][0]:.3f} to {m['lr_test_auc_95ci'][1]:.3f}")
        c3.metric("XGBoost PR-AUC", f"{m['xgb_test_pr_auc']:.3f}")
        c3.caption(f"Logistic Regression {m['lr_test_pr_auc']:.3f}")
        c4.metric("AUC lift over baseline", f"{m['auc_lift_pct']:.1f}%")
        c4.caption(f"95% CI {m['auc_lift_pct_95ci'][0]:.1f}% to {m['auc_lift_pct_95ci'][1]:.1f}%")

        st.subheader(f"At the decision threshold ({m['threshold']:.3f})")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Precision", f"{m['precision']:.1%}")
        d2.metric("Recall", f"{m['recall']:.1%}")
        d3.metric("F1", f"{m['f1']:.3f}")
        d4.metric("Accuracy", f"{m['accuracy']:.1%}")
        d4.caption(f"Always predicting 'no default' scores {m['majority_class_accuracy']:.1%}")
        st.dataframe(
            pd.DataFrame(
                m["confusion_matrix"],
                index=["Actual: no default", "Actual: default"],
                columns=["Predicted: no default", "Predicted: default"],
            )
        )

        with st.expander("Full metrics (JSON)"):
            st.json(m)
    else:
        st.info("Add your evaluation metrics as `models/metrics.json` to show them here.")

    figures = sorted(FIG_DIR.glob("*.png")) if FIG_DIR.exists() else []
    if figures:
        st.subheader("Figures")
        cols = st.columns(2)
        for i, path in enumerate(figures):
            cols[i % 2].image(str(path), caption=path.stem.replace("_", " ").title())
    elif not METRICS_PATH.exists():
        st.info("Add plots as PNG files in `models/figures/` to show them here.")
