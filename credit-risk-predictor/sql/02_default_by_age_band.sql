-- Default rate by age band.
SELECT
    CASE
        WHEN age < 25 THEN '1. Under 25'
        WHEN age < 35 THEN '2. 25-34'
        WHEN age < 45 THEN '3. 35-44'
        WHEN age < 55 THEN '4. 45-54'
        WHEN age < 65 THEN '5. 55-64'
        ELSE               '6. 65+'
    END                                             AS age_band,
    COUNT(*)                                        AS borrowers,
    SUM(defaulted)                                  AS defaults,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2)     AS default_rate_pct
FROM borrowers
WHERE age BETWEEN 18 AND 100
GROUP BY age_band
ORDER BY age_band;
