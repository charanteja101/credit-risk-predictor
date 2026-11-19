-- Credit Risk Predictor: database schema (SQLite syntax; also valid in PostgreSQL).
-- Load the Kaggle file with:  python sql/load_sqlite.py

DROP TABLE IF EXISTS borrowers;
CREATE TABLE borrowers (
    borrower_id            INTEGER PRIMARY KEY,
    defaulted              INTEGER NOT NULL,   -- 1 = serious delinquency within 2 years
    revolving_utilization  REAL,
    age                    INTEGER,
    past_due_30_59         INTEGER,
    debt_ratio             REAL,
    monthly_income         REAL,               -- NULL for ~20% of borrowers
    open_credit_lines      INTEGER,
    times_90_late          INTEGER,
    real_estate_loans      INTEGER,
    past_due_60_89         INTEGER,
    dependents             INTEGER             -- NULL for ~2.6% of borrowers
);

-- Optional: model scores for the held-out test set (exported from your own scoring code, see dashboard/README.md).
DROP TABLE IF EXISTS model_scores;
CREATE TABLE model_scores (
    row_id               INTEGER PRIMARY KEY,
    actual_default       INTEGER NOT NULL,
    default_probability  REAL NOT NULL,
    risk_band            TEXT NOT NULL,
    flagged_high_risk    INTEGER NOT NULL
);
