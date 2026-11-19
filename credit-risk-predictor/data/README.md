# Data

The raw data is not stored in this repository.

## Download

1. Go to https://www.kaggle.com/c/GiveMeSomeCredit/data (a free Kaggle account is needed).
2. Download `cs-training.csv`.
3. Place it here as `data/cs-training.csv`.

`cs-test.csv` has no labels, so it is not used.

## Schema (`cs-training.csv`)

| Column | Meaning |
|---|---|
| `SeriousDlqin2yrs` | Target: 1 if the borrower was 90+ days past due within 2 years |
| `RevolvingUtilizationOfUnsecuredLines` | Balance on credit cards and lines divided by credit limit |
| `age` | Borrower age in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30-59 days past due |
| `DebtRatio` | Monthly debt payments divided by monthly income |
| `MonthlyIncome` | Monthly income (about 20% missing) |
| `NumberOfOpenCreditLinesAndLoans` | Open loans and credit lines |
| `NumberOfTimes90DaysLate` | Times 90+ days late |
| `NumberRealEstateLoansOrLines` | Mortgage and real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60-89 days past due |
| `NumberOfDependents` | Dependents (about 2.6% missing) |

The first, unnamed column is a row index and is dropped.

## Known data issues (handled in `src/data.py`)

- 14 rows have an invalid age: one row with age 0 and 13 rows aged 101 to 109. Rows outside 18-100 are dropped.
- The values 96 and 98 in the three past-due columns are placeholder codes, not real counts.
- `RevolvingUtilizationOfUnsecuredLines` (max 50,708), `DebtRatio` (max 329,664) and `MonthlyIncome` (max 3,008,750) have extreme outliers; they are capped at 2, 5 and 50,000.
- `DebtRatio` is above 5 for 29,646 rows, and 92% of those rows have missing `MonthlyIncome`. When income is unknown the column looks like an absolute debt amount rather than a ratio. Capping it at 5 collapses these rows together; the `income_missing` feature keeps the information that income was unknown.
- The file contains 609 exact duplicate rows, removed before splitting.
- 269 rows have placeholder codes (96/98). Their default rate is 54.6%, so the codes carry signal and are kept as the `late_code_flag` feature.

After cleaning, 148,452 rows remain (default rate 6.73%).

Please follow the dataset's licence and competition rules on Kaggle.
