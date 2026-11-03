# Machine Failure Early-Warning: Assessment

Can we flag machine observations where a failure is likely within the next 7 days, early enough to inspect and with a manageable number of false alarms?

**Short answer:** there is a real, usable signal in the sensor data, but the two maintenance fields that make the model look excellent are leaking the answer. With them removed, the honest model is **good enough for a supervised pilot (shadow mode), not for unsupervised deployment.**

---

## 1. Summary for business stakeholders

| What we measured (held-out Jan to Feb 2025, never used for building) | Result |
|---|---|
| Failures caught (at least one alert in the 7 days before) | **38 of 44 (86%)**, 95% range 76% to 95% |
| Alerts raised across the 120-machine fleet | **~32 per week** (about 1.2 per machine per month) |
| Share of alerts that were real | **~1 in 5 (22%)**, 95% range 17% to 28% |
| Typical warning before failure | **~4 days** (median) |
| Simple rule "alert when vibration is high", same alert budget | catches only 32 of 44 (73%) |

**What this means**
- **Value:** the model catches about 6 more failures in 2 months than a vibration rule, for the same inspection effort.
- **Cost:** about 4 of every 5 alerts lead to an inspection that finds nothing urgent. Whether ~32 inspections a week is affordable is an operations decision. The threshold can be moved (see figure 5).
- **Misses:** 6 failures gave almost no warning in the sensors. No model on this data will catch those.

**Recommendation:** run a **3-month shadow pilot**: model scores every observation, planners see the alerts, decisions stay with engineers. Go-live only after the data questions in section 4 are answered and the pilot confirms the catch rate and alert volume.

---

## 2. Key findings (technical)

### Finding 1: two fields leak the target and must not be used
`corrective_orders_14d` and `inspection_interval_days` look like the best predictors (with them, PR-AUC jumps from 0.48 to 0.77). But their timing does not match their description:

![Leakage fingerprint](outputs/figures/01_leakage_fingerprint.png)

- A field that counts orders in the **previous** 14 days should only rise **after** a failure. Instead it rises symmetrically, peaking right at the failure, and is ~0 more than 14 days before.
- `inspection_interval_days` is identical to it before every failure (switches 30 to 14 days at the same moment) and stays at 14 for ~3 weeks after: it is a **reaction** to the failure.
- Most likely cause: orders are dated by the repair/failure date, or the field was rebuilt from the work-order system after the fact. At scoring time the information would not exist.
- **Decision:** excluded. The leaky model is kept in the results only to show how misleading it would be.

### Finding 2: three data quality problems, one of which silently breaks the model
![Pressure unit shift](outputs/figures/03_pressure_unit_shift.png)

| Issue | Rows | Fix |
|---|---|---|
| South site pressure reported in **psi** from 14 Dec 2024 (all 24 South machines, values ~6.9x too low) | 393 | Multiply by 6.895. Restores the exact prior distribution. |
| Exact duplicate observations (different `record_id`, identical everything else) | 80 | Dropped |
| `vibration_mm_s = -999` sentinel | 10 | Set to missing |
| `machine_class = "unknown"` on single rows of 15 machines | 15 | Filled with the machine's own class |
| Last 7 days: failures after the data extract are not visible (positive rate falls to 1.7%) | 171 | Excluded from training and evaluation |

Without the pressure fix, test PR-AUC falls from **0.48 to 0.08**. The unit change lands exactly in the test period, so a random split would have hidden it.

### Finding 3: the real signal is gradual drift that resets after repair
![Sensor signals](outputs/figures/02_sensor_signals.png)

Vibration, bearing temperature and motor current rise, and discharge pressure falls, over the weeks before a failure, then reset after the repair. That is the pattern a maintenance engineer would expect, which gives confidence the model is learning something physical.

### Finding 4: model performance on the untouched test period
![Precision recall](outputs/figures/04_precision_recall_test.png)
![Trade-off](outputs/figures/05_alerts_vs_failures_caught.png)

- Logistic regression and gradient boosting perform within noise of each other. **Logistic regression was chosen**: it needed fewer alerts for the same catch rate on validation, and it is easier to explain and monitor.
- Most important inputs: discharge pressure, unplanned stops, vibration, machine class, bearing temperature above ambient.

![Importance](outputs/figures/06_feature_importance.png)

---

## 3. Approach and key decisions

| Decision | Why |
|---|---|
| **Unit of prediction = one observation**, scored using only that observation and the same machine's earlier ones | The brief says each record is scored the moment it is recorded. All history features use past data only. |
| **Time-based split with a 7-day gap** (walk-forward validation Jul to Dec 2024, test Jan to Feb 2025) | The label looks 7 days ahead, so a random split leaks future information. Machines are the same in production, so we test on the future, not on new machines. |
| **Success metric = share of failure events caught vs alerts per week** | That is how operations will experience the tool. Observation-level PR-AUC is reported too. ROC-AUC looks flattering at a 5% base rate, so it is secondary. |
| **Threshold set on validation** to catch at least 80% of failures, then frozen and applied to test | Missing a failure costs more than an inspection, but alert volume must stay practical. 80% is a starting point for the business to adjust. |
| **Few, fixed models** (vibration rule, logistic regression, gradient boosting), no heavy tuning | ~340 failures in total; tuning many settings would mostly fit noise. |
| **Uncertainty by resampling whole machines** | Observations of one machine are correlated; row-level resampling would overstate confidence. |
| **Failure events reconstructed from the label** (a run of positive labels = one failure) | Needed to measure "failures caught" and lead time. Only used for evaluation, never as a model input. |

Engineered features: each sensor vs the machine's own recent baseline (median of its previous 8 observations), change since the previous observation, 3-observation vibration slope, bearing temperature above ambient, current per unit load, missing-reading flags.

---

## 4. Assumptions and questions for the data owner
Full list in [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md). The ones that change the answer:

1. **Corrective orders and inspection interval:** when exactly is each order timestamped (raised, scheduled, completed)? Was the 14-day count computed live or rebuilt later? (We assume rebuilt or failure-dated, hence excluded.)
2. **Unplanned stops:** is the failure itself counted as a stop, and are stops logged in real time? It is the second most important input. Without it the model still works but needs ~44 instead of ~32 alerts/week for the same catch rate.
3. **Pressure units:** confirm the South feed switched to psi on 14 Dec 2024, and add a unit check upstream.
4. **Failure definition:** what counts as a "failure" in the records, and are failures between observations always captured?

---

## 5. How to run

Requires Python 3.10+.

```bash
git clone <this repo>
cd machine-failure-early-warning
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_analysis.py
```

Runs in about 1 to 2 minutes and regenerates everything in `outputs/`:
- `metrics.json`: all numbers quoted in this README
- `validation_model_comparison.csv`: walk-forward comparison of all candidates
- `figures/`: the charts above

The data file is included at `data/machine_observations.csv`. Random seeds are fixed, so results are identical on re-run.

### Repository layout
```
run_analysis.py          end-to-end pipeline (clean -> features -> validate -> test -> figures)
src/data.py              loading, cleaning, failure-event reconstruction
src/features.py          past-only feature engineering, feature sets, excluded fields
src/models.py            candidate models
src/evaluate.py          observation- and event-level metrics, machine bootstrap
docs/ASSUMPTIONS.md      assumptions and questions for the data owner
docs/PRESENTATION.md     10-minute talk track
outputs/                 generated results
```

---

## 6. Limitations and next steps

**Limitations**
- Test window is 2 months with 44 failures; ranges above are wide. South had only 3 test failures.
- Precision is counted per observation; consecutive alerts on one machine are often the same case, so the real inspection load is somewhat lower than ~32/week.
- Lead time is capped by the 7-day label definition.
- Failure times are estimated from the label (within ~4 days on average).

**Before deployment**
1. Resolve the data questions in section 4.
2. Shadow pilot for ~3 months; track failures caught, alert volume, and what inspections actually found.
3. Agree the threshold with operations using a cost per missed failure vs cost per inspection.
4. Add input monitoring (units, ranges, missing rates) so a change like the psi switch raises an alarm instead of silently degrading the model.
5. Then try: machine-level alert de-duplication, longer history features, and a cost-based threshold.
