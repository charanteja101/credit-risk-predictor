import argparse
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

COLUMN_MAP = {
    "Unnamed: 0": "borrower_id",
    "SeriousDlqin2yrs": "defaulted",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "times_90_late",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
    "NumberOfDependents": "dependents",
}


def main():
    ap = argparse.ArgumentParser(description="Load cs-training.csv (and optional model scores) into SQLite")
    ap.add_argument("--data", default=str(ROOT / "data" / "cs-training.csv"))
    ap.add_argument("--scores", default=str(ROOT / "dashboard" / "powerbi_data" / "scored_test.csv"))
    ap.add_argument("--db", default=str(ROOT / "sql" / "credit_risk.db"))
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.executescript((ROOT / "sql" / "00_schema.sql").read_text())

    df = pd.read_csv(args.data).rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]
    df.to_sql("borrowers", con, if_exists="append", index=False)
    print(f"borrowers: {len(df):,} rows loaded")

    if Path(args.scores).exists():
        s = pd.read_csv(args.scores)[
            ["row_id", "actual_default", "default_probability", "risk_band", "flagged_high_risk"]
        ]
        s.to_sql("model_scores", con, if_exists="append", index=False)
        print(f"model_scores: {len(s):,} rows loaded")
    else:
        print("model_scores: skipped (export your scored test set first, see dashboard/README.md)")

    con.commit()
    con.close()
    print(f"Database written to {args.db}")
    print("Run a query, e.g.:  sqlite3 sql/credit_risk.db < sql/02_default_by_age_band.sql")


if __name__ == "__main__":
    main()
