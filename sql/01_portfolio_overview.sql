-- Portfolio size, default rate and data-quality counts.
SELECT
    COUNT(*)                                                    AS borrowers,
    SUM(defaulted)                                              AS defaults,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2)                 AS default_rate_pct,
    ROUND(AVG(age), 1)                                          AS avg_age,
    ROUND(AVG(monthly_income), 0)                               AS avg_monthly_income,
    SUM(CASE WHEN monthly_income IS NULL THEN 1 ELSE 0 END)     AS missing_income,
    SUM(CASE WHEN dependents IS NULL THEN 1 ELSE 0 END)         AS missing_dependents,
    SUM(CASE WHEN past_due_30_59 >= 96 OR past_due_60_89 >= 96 OR times_90_late >= 96
             THEN 1 ELSE 0 END)                                 AS placeholder_code_rows
FROM borrowers;
