# Assumptions and questions for the data owner

Each item: what we saw, what we assumed, what to verify.

## 1. Scoring moment
- **Assumed:** a record is scored once, at `observation_timestamp`, and all its fields are known at that moment.
- **Consequence:** features use only the current and earlier observations of the same machine.
- **Verify:** is each field populated at observation time, or filled in later by a batch job?

## 2. Field timing and lineage

| Field | What we saw | Assumption | Used? | Verify with data owner |
|---|---|---|---|---|
| `corrective_orders_14d` | Rises symmetrically around the failure: ~90% within 2 days either side of a failure, ~0 more than 14 days before. A "previous 14 days" count should only rise after a failure. 44% of rows with a value of 1 are positive. | Orders are dated by the failure/repair date, or the count was rebuilt from the work-order system after the fact. Leaks the target. | **No** | Which date is used (created, scheduled, completed)? Is the count computed live or reconstructed? Can orders be backdated? |
| `inspection_interval_days` | Only 14 or 30. Before a failure it switches to 14 exactly when a corrective order appears (identical rates); stays 14 for ~3 weeks after. | A policy reaction to the failure. Leaks the target. | **No** | When is the change recorded vs when it takes effect? Is the history of the interval kept, or overwritten? |
| `unplanned_stops_14d` | Rises gradually over 3 weeks before failure, peaks at failure, decays after. | Logged in real time; the gradual pre-failure rise is a genuine warning sign. | **Yes** (with sensitivity check) | Is the failure itself counted as a stop? Any logging delay? |
| Sensor readings | Drift before failure, reset after repair. | Recorded at the observation time. | Yes | Are readings instantaneous or averaged over a window? |
| `fails_within_7d` | 339 failure episodes reconstructed; last week positive rate 1.7% vs ~5% normally. | Failures after the extract are not yet known; last 7 days excluded. | Target | How is a "failure" defined and logged? Can a failure be missed if it happens between observations? |

**Sensitivity check:** removing `unplanned_stops_14d` keeps the same 86% catch rate on test but needs ~44 instead of ~32 alerts per week. The conclusion does not depend on it.

## 3. Data quality

| Issue | Assumption / fix | Verify |
|---|---|---|
| South `discharge_pressure_kpa` ~6.9x too low from 2024-12-14 16:23 (all 24 South machines, no other site) | psi written into a kPa column; multiplied by 6.894757. The converted values match the pre-change distribution (median 503 vs 501 kPa). | Confirm the source change; add a unit check in the pipeline. |
| 80 exact duplicates (same machine, time and readings, new `record_id`) | Double ingestion; keep one. | Why are records duplicated? |
| Near-duplicates: some machines have two readings minutes apart with slightly different values | Genuine re-reads; kept. | Is this a re-measurement process? |
| `vibration_mm_s = -999` (10 rows) | Sensor error code; treated as missing. | Other sentinel codes? |
| ~1% missing vibration and bearing temperature | Kept as missing with a flag; models handle it. Missing vibration rows have a slightly higher failure rate (7.8%). | Does a sensor dropout itself signal a problem? |
| `machine_class = "unknown"` on one row each for 15 machines | Class is fixed per machine; filled from the machine's other rows. | Can a machine change class? |
| January 2024 positive rate is low (0.8%) | Start-of-data effect; kept. | Was failure logging complete from the start? |

## 4. Evaluation assumptions
- **Fleet is stable:** the same 120 machines are scored in production, so we split by time, not by machine.
- **Embargo of 7 days** between training and evaluation data, because the label looks 7 days ahead.
- **Catching a failure** = at least one alert among the observations in the 7 days before it. One timely alert is enough to trigger an inspection.
- **80% catch-rate target** is a placeholder. The right threshold depends on the cost of a missed failure vs an inspection, which the business should provide.
