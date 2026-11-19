-- Default rate by income quartile (borrowers with known income). Uses a window function (SQLite 3.25+).
WITH ranked AS (
    SELECT
        defaulted,
        monthly_income,
        NTILE(4) OVER (ORDER BY monthly_income) AS income_quartile
    FROM borrowers
    WHERE monthly_income IS NOT NULL
)
SELECT
    income_quartile,
    ROUND(MIN(monthly_income), 0)                   AS min_income,
    ROUND(MAX(monthly_income), 0)                   AS max_income,
    COUNT(*)                                        AS borrowers,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2)     AS default_rate_pct
FROM ranked
GROUP BY income_quartile
ORDER BY income_quartile;
