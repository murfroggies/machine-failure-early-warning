# 10-minute talk track

Present straight from the README. Timings are a guide.

## 1. The ask and how I framed it (1 min)
- Flag observations where a machine is likely to fail in the next 7 days.
- Missing a failure costs more than an inspection, but too many alarms kills adoption.
- So I measure success as: **share of failures caught vs alerts per week**, not accuracy.

## 2. First, can we trust the data? (3 min) — figures 1 and 3
- Two fields looked like the best predictors. Plotting them against time-to-failure showed they **peak at the failure itself**, which a "previous 14 days" count cannot do. They describe the response to a failure, not a warning. Removed.
- Using them would have reported PR-AUC 0.77 instead of 0.48: a model that looks great and fails in production.
- South pressure switched to psi in December, exactly inside the test period. Unfixed, the model collapses (PR-AUC 0.08). Lesson: needs input monitoring.
- Smaller issues: duplicates, a -999 code, unknown class, the last 7 days' labels are incomplete.

## 3. What is the real signal? (1 min) — figure 2
- Vibration, bearing temperature and current creep up, pressure falls, then everything resets after repair. Physically sensible.

## 4. How I evaluated (1.5 min)
- Split by time with a 7-day gap: train on the past, test on Jan to Feb 2025.
- Walk-forward validation for choosing the model and threshold; the test set was used once.
- Compared to a simple vibration rule so "better than what we could do without ML" is measurable.
- Uncertainty by resampling whole machines.

## 5. Results (2 min) — figures 4 and 5
- 86% of failures caught (38 of 44), ~4 days of warning, ~32 alerts per week, about 1 in 5 alerts real.
- Vibration rule with the same alert budget: 73%.
- Logistic regression chosen over gradient boosting: same performance, fewer alerts, easier to explain.
- 6 misses had almost no sensor drift: a data limit, not a modelling one.

## 6. Recommendation (1.5 min)
- **Not ready for unsupervised deployment. Ready for a 3-month shadow pilot.**
- Conditions: answer the data owner questions (order timing, stops logging, units), agree the threshold with operations using real costs, add input monitoring.

## Likely questions
- *Why not keep the corrective-order field if it is so predictive?* Because it is only predictive in hindsight. If the data owner shows it is genuinely recorded before the failure, I would test it again with a strict as-of extract.
- *Why not a random or machine split?* The same machines are scored in production, and the label looks ahead 7 days, so the honest test is the future.
- *Why logistic regression?* Within noise of boosting on all folds; fewer alerts on validation; coefficients are easy to audit.
- *Why 80%?* A placeholder; figure 5 shows the whole trade-off for operations to choose from.
- *What would you do with more time?* Alert de-duplication per machine, cost-based threshold, longer history features, test on a longer period.
