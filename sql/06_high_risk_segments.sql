-- Segments with the highest default rates (at least 200 borrowers so the rates are stable).
SELECT
    CASE WHEN age < 35 THEN 'Under 35' ELSE '35+' END                AS age_group,
    CASE WHEN revolving_utilization >= 1 THEN 'Over limit'
         WHEN revolving_utilization >= 0.6 THEN 'High utilisation'
         ELSE 'Normal utilisation' END                               AS utilisation_group,
    CASE WHEN times_90_late > 0 THEN '90+ days late before'
         ELSE 'No 90+ day lates' END                                 AS late_history,
    COUNT(*)                                                         AS borrowers,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2)                      AS default_rate_pct
FROM borrowers
WHERE age BETWEEN 18 AND 100
GROUP BY age_group, utilisation_group, late_history
HAVING COUNT(*) >= 200
ORDER BY default_rate_pct DESC
LIMIT 10;
