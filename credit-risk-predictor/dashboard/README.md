# Power BI dashboard

A `.pbix` file cannot be generated from code, so this folder gives you the data contract, the DAX and a layout guide. Build the report in about 30 minutes, then save screenshots here.

## 1. Export your scored test set

`python -m src.train` writes `dashboard/powerbi_data/scored_test.csv` automatically. If you only want to rebuild it from the saved model without retraining, run:

```bash
python -m src.export_scores --data data/cs-training.csv
```

The folder is git-ignored. The file has one row per held-out borrower (29,691 rows) and these columns:

| Column | Meaning |
|---|---|
| `row_id` | Row index of the cleaned dataset (unique) |
| `age` | Borrower age |
| `RevolvingUtilizationOfUnsecuredLines` | Credit utilisation (1.0 = at limit) |
| `actual_default` | 1 if the borrower defaulted, else 0 |
| `default_probability` | Model's predicted probability |
| `risk_band` | Low (below half the threshold), Medium (between half the threshold and the threshold) or High (at or above the threshold) |
| `flagged_high_risk` | 1 if the probability is at or above the decision threshold (High band), else 0 |

You can add any other columns you want to slice by.

## 2. Build the report in Power BI Desktop

1. Get Data -> Text/CSV -> load `scored_test.csv` and name the table `scored_test`.
2. Add the calculated columns and measures from `measures.dax`. Set `risk_band` to *Sort by column* -> `risk_band_order` so the bands show as Low, Medium, High instead of alphabetically.
3. Suggested layout:
   - **Page 1, Portfolio risk:** cards for Total Borrowers, Actual Default Rate and Flagged High Risk; a column chart of Actual Default Rate by `age_band`; a column chart of Actual Default Rate by `utilisation_band`; a donut of `risk_band`.
   - **Page 2, Model performance:** cards for Precision and Recall; a column chart of Actual Default Rate by `risk_band` (it should rise from low to high risk); a bar chart of `outcome`.
   - Slicers: `age_band`, `utilisation_band`, `risk_band`.
4. Save as `dashboard/credit_risk_dashboard.pbix`.

The DAX has not been tested in Power BI Desktop, so check the column and table names match your CSV.

## 3. Screenshots

Export each page as an image and save them in `dashboard/screenshots/` so reviewers can see the dashboard without Power BI. Reference them in the main README.
