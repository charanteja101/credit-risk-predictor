-- Default rate by past delinquency. Values 96 and 98 are placeholder codes, so they get their own group.
SELECT
    CASE
        WHEN past_due_30_59 >= 96 OR past_due_60_89 >= 96 OR times_90_late >= 96 THEN '4. Placeholder code (96/98)'
        WHEN past_due_30_59 + past_due_60_89 + times_90_late = 0 THEN '1. Never late'
        WHEN past_due_30_59 + past_due_60_89 + times_90_late <= 2 THEN '2. 1-2 late events'
        ELSE '3. 3+ late events'
    END                                             AS delinquency_history,
    COUNT(*)                                        AS borrowers,
    SUM(defaulted)                                  AS defaults,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2)     AS default_rate_pct
FROM borrowers
GROUP BY delinquency_history
ORDER BY delinquency_history;
