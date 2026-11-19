-- Default rate by revolving credit utilisation (default rate rises steeply with utilisation).
SELECT
    CASE
        WHEN revolving_utilization < 0.1 THEN '1. Under 10%'
        WHEN revolving_utilization < 0.3 THEN '2. 10-30%'
        WHEN revolving_utilization < 0.6 THEN '3. 30-60%'
        WHEN revolving_utilization < 1.0 THEN '4. 60-100%'
        ELSE                                  '5. Over limit (100%+)'
    END                                             AS utilisation_band,
    COUNT(*)                                        AS borrowers,
    SUM(defaulted)                                  AS defaults,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2)     AS default_rate_pct
FROM borrowers
GROUP BY utilisation_band
ORDER BY utilisation_band;
