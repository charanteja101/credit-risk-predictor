-- Model lift: do higher predicted risk deciles really default more? (needs the model_scores table)
WITH deciles AS (
    SELECT
        actual_default,
        default_probability,
        NTILE(10) OVER (ORDER BY default_probability DESC) AS risk_decile
    FROM model_scores
),
overall AS (
    SELECT 1.0 * SUM(actual_default) / COUNT(*) AS base_rate FROM model_scores
)
SELECT
    risk_decile,
    COUNT(*)                                                          AS borrowers,
    SUM(actual_default)                                               AS defaults,
    ROUND(100.0 * SUM(actual_default) / COUNT(*), 2)                  AS default_rate_pct,
    ROUND((1.0 * SUM(actual_default) / COUNT(*)) / (SELECT base_rate FROM overall), 2) AS lift_vs_average
FROM deciles
GROUP BY risk_decile
ORDER BY risk_decile;
