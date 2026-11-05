"""Build the interactive findings page: report/index.html

    python run_analysis.py          # first, writes outputs/metrics.json
    python report/build_report.py   # then, writes report/index.html

Recomputes the chart data with the same pipeline and the threshold chosen on
validation (read from outputs/metrics.json), then injects it into
report/template.html. The page is a single self-contained file.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_analysis import EMBARGO, TEST_START, fit_predict  # noqa: E402
from src.data import TARGET, TS, attach_event_info, clean, load_raw, reconstruct_failures  # noqa: E402
from src.evaluate import at_threshold, obs_metrics, sweep  # noqa: E402
from src.features import build_features, feature_sets  # noqa: E402

warnings.filterwarnings("ignore")


def r(x, n=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def binned(df, cols, width=2, lim=28):
    b = pd.cut(df["days_to_nearest_failure"], np.arange(-lim, lim + 1, width))
    t = df.groupby(b, observed=True)[cols].mean()
    return [{"d": r(i.mid, 1), **{c: r(t.loc[i, c]) for c in cols}} for i in t.index]


def main():
    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    thr = metrics["test"]["threshold_from_validation"]
    chosen_model = metrics["test"]["chosen_model"]
    chosen_feats = metrics["test"]["chosen_features"]

    raw = load_raw(ROOT / "data" / "machine_observations.csv")
    df, _ = clean(raw)
    ev = reconstruct_failures(df)
    df = attach_event_info(df, ev).merge(ev[["event_id", "failure_est"]], on="event_id", how="left")
    df = build_features(df)
    df = df[df[TS] <= df[TS].max() - EMBARGO].copy()

    fs = feature_sets()
    test = df[df[TS] >= pd.Timestamp(TEST_START)].copy()
    train = df[df[TS] < pd.Timestamp(TEST_START) - EMBARGO]
    alt = "gradient_boosting" if chosen_model == "logistic_regression" else "logistic_regression"
    _, test["chosen"] = fit_predict(chosen_model, fs[chosen_feats], train, test)
    _, test["alternative"] = fit_predict(alt, fs[chosen_feats], train, test)
    _, test["leaky"] = fit_predict("gradient_boosting", fs["leaky_all_fields"], train, test)
    test["rule"] = test["vibration_mm_s"].fillna(train["vibration_mm_s"].median())

    # Threshold sweep (model + rule) for the interactive slider
    sw = sweep(test, "chosen", n=300)
    sw = pd.concat([sw, pd.DataFrame([at_threshold(test, "chosen", thr)])])  # exact recommended point
    sw = sw.drop_duplicates("alerts_per_week", keep="last").sort_values("threshold")
    rule_sw = sweep(test, "rule", n=300).drop_duplicates("alerts_per_week")
    keep = ["threshold", "event_recall", "events_caught", "events_total", "obs_precision",
            "alerts_per_week", "false_alerts_per_week", "median_lead_days"]
    sweep_rows = [{k: r(row[k]) for k in keep} for _, row in sw.iterrows()]
    rule_rows = [{"alerts_per_week": r(a, 2), "event_recall": r(e)} for a, e in
                 zip(rule_sw.alerts_per_week, rule_sw.event_recall)]

    # Precision-recall curves, thinned
    pr = {}
    for c in ["leaky", "chosen", "alternative", "rule"]:
        p, rc, _ = precision_recall_curve(test[TARGET], test[c])
        idx = np.unique(np.linspace(0, len(p) - 1, 120).astype(int))
        pr[c] = {"auc": r(obs_metrics(test[TARGET], test[c])["pr_auc"], 6),
                 "pts": [[r(rc[i]), r(p[i])] for i in idx]}

    # Leakage fingerprint + sensor drift (all clean data, not just test)
    d = df.assign(interval_14=(df.inspection_interval_days == 14).astype(float))
    leak = binned(d, ["corrective_orders_14d", "interval_14"])
    sensors = binned(df, ["vibration_mm_s", "bearing_temp_c", "motor_current_a",
                          "discharge_pressure_kpa", "unplanned_stops_14d"])

    # South pressure, raw, daily median (to show the unit switch)
    s = raw[raw.site == "South"].set_index(TS)["discharge_pressure_kpa"].resample("3D").median().dropna()
    pressure = [[str(t.date()), r(v, 1)] for t, v in s.items()]

    imp = pd.read_csv(ROOT / "outputs" / "permutation_importance_test.csv", index_col=0).iloc[:, 0]
    importance = [[k, r(v)] for k, v in imp.sort_values(ascending=False).items()]

    data = {
        "chosen_model": chosen_model,
        "threshold": r(thr, 4),
        "metrics": metrics,
        "sweep": sweep_rows,
        "rule_sweep": rule_rows,
        "pr": pr,
        "leak": leak,
        "sensors": sensors,
        "pressure": pressure,
        "importance": importance,
        "test_counts": {"rows": int(len(test)), "machines": int(test.machine_id.nunique()),
                        "events": int(test.loc[test[TARGET] == 1, "event_id"].nunique()),
                        "start": str(test[TS].min().date()), "end": str(test[TS].max().date())},
        "op": {k: r(v) for k, v in at_threshold(test, "chosen", thr).items()},
    }
    tpl = (ROOT / "report" / "template.html").read_text()
    out = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (ROOT / "report" / "index.html").write_text(out)
    print(f"wrote report/index.html ({len(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
