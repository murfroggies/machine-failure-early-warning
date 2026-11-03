"""End-to-end, reproducible analysis.

    python run_analysis.py

Writes everything to outputs/ (metrics.json, tables, figures). Runtime < 2 min.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve

from src.data import TARGET, TS, attach_event_info, clean, load_raw, reconstruct_failures
from src.evaluate import (
    at_threshold,
    machine_bootstrap,
    obs_metrics,
    sweep,
    threshold_for_event_recall,
)
from src.features import build_features, feature_sets
from src.models import MODELS

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

EMBARGO = pd.Timedelta(days=7)  # label looks 7 days ahead -> purge gap
EVENT_RECALL_TARGET = 0.80      # business: missing a failure is the costly error
# Walk-forward validation windows (model selection + threshold), then a
# final untouched test window.
VAL_WINDOWS = [
    ("2024-07-01", "2024-09-01"),
    ("2024-09-01", "2024-11-01"),
    ("2024-11-01", "2025-01-01"),
]
TEST_START = "2025-01-01"

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titleweight": "bold"})
C = {"gradient_boosting": "#1f4e9c", "logistic_regression": "#3a9d8f",
     "vibration_rule": "#999999", "leaky": "#c0392b"}


def fit_predict(model_name, feats, train, score_df):
    model = MODELS[model_name](feats)
    model.fit(train[feats], train[TARGET])
    return model, model.predict_proba(score_df[feats])[:, 1]


def main():
    # ------------------------------------------------------------------ data
    raw = load_raw(ROOT / "data" / "machine_observations.csv")
    df, clean_log = clean(raw)
    events = reconstruct_failures(df)
    df = attach_event_info(df, events)
    df = df.merge(events[["event_id", "failure_est"]], on="event_id", how="left")
    df = build_features(df)

    # Right-censoring: failures after the extract date are unobserved, so the
    # last 7 days' labels are unreliable (positive rate drops to ~1.7%).
    data_end = df[TS].max()
    censored = df[TS] > data_end - EMBARGO
    clean_log["censored_last_7d_rows_excluded"] = int(censored.sum())
    df = df[~censored].copy()
    clean_log["failure_events_reconstructed"] = int(len(events))

    fsets = feature_sets()

    # -------------------------------------------- leakage fingerprint figure
    leak_fig(df)
    sensor_fig(df)
    pressure_fig(raw)

    # ------------------------------------------------ walk-forward validation
    oof = []
    for k, (vs, ve) in enumerate(VAL_WINDOWS):
        vs, ve = pd.Timestamp(vs), pd.Timestamp(ve)
        train = df[df[TS] < vs - EMBARGO]
        val = df[(df[TS] >= vs) & (df[TS] < ve)].copy()
        val["fold"] = k
        for fs in ["sensors_plus_history", "sensors_only", "leaky_all_fields"]:
            for m in MODELS:
                if fs == "leaky_all_fields" and m != "gradient_boosting":
                    continue
                _, val[f"{m}|{fs}"] = fit_predict(m, fsets[fs], train, val)
        val["vibration_rule|-"] = val["vibration_mm_s"].fillna(val["vibration_mm_s"].median())
        oof.append(val)
    oof = pd.concat(oof)
    score_cols = [c for c in oof.columns if "|" in c]

    rows = []
    for c in score_cols:
        per_fold = [obs_metrics(g[TARGET], g[c])["pr_auc"] for _, g in oof.groupby("fold")]
        thr = threshold_for_event_recall(oof, c, EVENT_RECALL_TARGET)
        op = at_threshold(oof, c, thr)
        m, fs = c.split("|")
        rows.append({
            "model": m, "features": fs,
            "pr_auc_mean": np.mean(per_fold), "pr_auc_sd": np.std(per_fold),
            "roc_auc": obs_metrics(oof[TARGET], oof[c])["roc_auc"],
            "precision_at_80pct_event_recall": op["obs_precision"],
            "alerts_per_week_at_80pct_event_recall": op["alerts_per_week"],
        })
    val_table = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    val_table.to_csv(OUT / "validation_model_comparison.csv", index=False)

    # Choose model on validation, excluding the leaky illustration.
    # Criterion = the business metric: fewest alerts per week while catching
    # >= 80% of failures. (PR-AUC differences between the two learned models
    # are within fold-to-fold noise; the simpler model wins ties.)
    honest = val_table[val_table.features != "leaky_all_fields"]
    best = honest.sort_values("alerts_per_week_at_80pct_event_recall").iloc[0]
    best_col = f"{best.model}|{best.features}"
    threshold = threshold_for_event_recall(oof, best_col, EVENT_RECALL_TARGET)

    # ------------------------------------------------------- final test
    test = df[df[TS] >= pd.Timestamp(TEST_START)].copy()
    train = df[df[TS] < pd.Timestamp(TEST_START) - EMBARGO]
    final_model, test["chosen"] = fit_predict(best.model, fsets[best.features], train, test)
    alt = "gradient_boosting" if best.model == "logistic_regression" else "logistic_regression"
    _, test["alternative"] = fit_predict(alt, fsets["sensors_plus_history"], train, test)
    _, test["leaky"] = fit_predict("gradient_boosting", fsets["leaky_all_fields"], train, test)
    test["vibration_rule"] = test["vibration_mm_s"].fillna(train["vibration_mm_s"].median())

    test_res = {
        "chosen_model": best.model, "chosen_features": best.features,
        "threshold_from_validation": threshold,
        "observation_level": {c: obs_metrics(test[TARGET], test[c])
                              for c in ["chosen", "alternative", "vibration_rule", "leaky"]},
        "alternative_model": alt,
        "operating_point": at_threshold(test, "chosen", threshold),
        "operating_point_ci95_by_machine_bootstrap": machine_bootstrap(test, "chosen", threshold),
        "rule_at_same_alert_volume": None,
    }
    # fair comparison: vibration rule given the same alert budget
    q = 1 - test_res["operating_point"]["alert_rate"]
    rule_thr = float(np.quantile(test["vibration_rule"], q))
    test_res["rule_at_same_alert_volume"] = at_threshold(test, "vibration_rule", rule_thr)

    # Missed failures: what did they look like?
    pos = test[test[TARGET] == 1].assign(alert=lambda d: d["chosen"] >= threshold)
    ev_caught = pos.groupby("event_id")["alert"].max()
    missed_ids = ev_caught[~ev_caught].index
    miss = pos[pos.event_id.isin(missed_ids)]
    hit = pos[~pos.event_id.isin(missed_ids)]
    cols = ["vibration_mm_s_vs_baseline", "bearing_temp_c_vs_baseline", "motor_current_a_vs_baseline", "unplanned_stops_14d"]
    test_res["missed_vs_caught_profile"] = {
        "missed": miss[cols].mean().round(2).to_dict(), "caught": hit[cols].mean().round(2).to_dict(),
        "n_missed_events": int(len(missed_ids)),
    }
    # Per-site stability (pressure fix matters for South)
    test_res["by_site"] = {
        s: {**obs_metrics(g[TARGET], g["chosen"]), **{k: v for k, v in at_threshold(g, "chosen", threshold).items()
                                                       if k in ("event_recall", "obs_precision", "events_total")}}
        for s, g in test.groupby("site", observed=True)
    }

    # Sensitivity: drop unplanned_stops_14d (its timing is unconfirmed).
    so_col = f"{best.model}|sensors_only"
    so_thr = threshold_for_event_recall(oof, so_col, EVENT_RECALL_TARGET)
    _, test["sensors_only"] = fit_predict(best.model, fsets["sensors_only"], train, test)
    test_res["sensitivity_without_unplanned_stops"] = {
        **obs_metrics(test[TARGET], test["sensors_only"]),
        **{k: v for k, v in at_threshold(test, "sensors_only", so_thr).items()
           if k in ("event_recall", "obs_precision", "alerts_per_week")},
    }

    # Sensitivity: what if the South pressure units had NOT been fixed?
    test_res["sensitivity_no_pressure_fix"] = pressure_sensitivity(raw, best, fsets, threshold)

    pr_fig(test, threshold, best.model, alt)
    tradeoff_fig(test, threshold)
    importance_fig(final_model, test, fsets[best.features])

    out = {"cleaning": clean_log, "validation_table": val_table.round(4).to_dict("records"),
           "test": test_res,
           "split": {"val_windows": VAL_WINDOWS, "test_start": TEST_START,
                     "test_end": str(test[TS].max()), "embargo_days": 7,
                     "n_train_final": int(len(train)), "n_test": int(len(test)),
                     "test_events": int(test.loc[test[TARGET] == 1, "event_id"].nunique())}}
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


def pressure_sensitivity(raw, best, fsets, threshold):
    from src.data import clean as _clean
    import src.data as d
    old = d.PRESSURE_UNIT_THRESHOLD_KPA
    d.PRESSURE_UNIT_THRESHOLD_KPA = -1  # disables the fix
    try:
        df2, _ = _clean(raw)
    finally:
        d.PRESSURE_UNIT_THRESHOLD_KPA = old
    ev = reconstruct_failures(df2)
    df2 = attach_event_info(df2, ev).merge(ev[["event_id", "failure_est"]], on="event_id", how="left")
    df2 = build_features(df2)
    df2 = df2[df2[TS] <= df2[TS].max() - EMBARGO]
    test = df2[df2[TS] >= pd.Timestamp(TEST_START)].copy()
    train = df2[df2[TS] < pd.Timestamp(TEST_START) - EMBARGO]
    _, test["s"] = fit_predict(best.model, fsets[best.features], train, test)
    south = test[test.site == "South"]
    return {"all_sites": {**obs_metrics(test[TARGET], test["s"]),
                          "event_recall": at_threshold(test, "s", threshold)["event_recall"],
                          "alerts_per_week": at_threshold(test, "s", threshold)["alerts_per_week"]},
            "south": {**obs_metrics(south[TARGET], south["s"]),
                      "event_recall": at_threshold(south, "s", threshold)["event_recall"],
                      "alerts_per_week": at_threshold(south, "s", threshold)["alerts_per_week"]}}


# ---------------------------------------------------------------- figures
def _binned(df, cols, width=2, lim=28):
    b = pd.cut(df["days_to_nearest_failure"], np.arange(-lim, lim + 1, width))
    out = df.groupby(b, observed=True)[cols].mean()
    out.index = [i.mid for i in out.index]
    return out


def leak_fig(df):
    d = df.assign(interval_14=(df.inspection_interval_days == 14).astype(float))
    t = _binned(d, ["corrective_orders_14d", "interval_14", TARGET])
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.axvspan(0, 7, color="#f4d03f", alpha=0.25, label="label = 1 zone (failure within 7 days)")
    ax.plot(t.index, t["corrective_orders_14d"], "-o", ms=3, color=C["leaky"], label="corrective order in 'previous 14 days'")
    ax.plot(t.index, t["interval_14"], "--", color="#7d3c98", label="inspection interval = 14 days")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("days until the nearest failure  (negative = after the failure)")
    ax.set_ylabel("share of observations")
    ax.set_title("Leakage fingerprint: maintenance fields peak around the failure itself")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "01_leakage_fingerprint.png"); plt.close(fig)


def sensor_fig(df):
    cols = ["vibration_mm_s", "bearing_temp_c", "motor_current_a", "discharge_pressure_kpa", "unplanned_stops_14d"]
    t = _binned(df, cols)
    fig, axs = plt.subplots(1, 5, figsize=(13, 2.8))
    for ax, c in zip(axs, cols):
        ax.axvspan(0, 7, color="#f4d03f", alpha=0.25)
        ax.plot(t.index, t[c], color="#1f4e9c")
        ax.axvline(0, color="k", lw=0.8); ax.invert_xaxis()
        ax.set_title(c, fontsize=8.5); ax.set_xlabel("days to failure")
    fig.suptitle("Genuine early-warning signals: readings drift before failure (pressure falls) and reset after repair", fontsize=9.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "02_sensor_signals.png"); plt.close(fig)


def pressure_fig(raw):
    s = raw[raw.site == "South"]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.scatter(s[TS], s.discharge_pressure_kpa, s=2, alpha=0.4, color="#555")
    ax.set_ylabel("discharge_pressure_kpa (raw)")
    ax.set_title("South site pressure drops ~6.9x on 14 Dec 2024: a psi feed in a kPa column")
    fig.tight_layout(); fig.savefig(FIG / "03_pressure_unit_shift.png"); plt.close(fig)


def pr_fig(test, thr, chosen_name, alt_name):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    nice = lambda n: n.replace("_", " ")
    lab = {"leaky": "Leaky fields included (NOT valid)", "chosen": f"Chosen: {nice(chosen_name)}",
           "alternative": f"Alternative: {nice(alt_name)}", "vibration_rule": "Vibration threshold rule"}
    col = {"leaky": C["leaky"], "chosen": C[chosen_name], "alternative": C[alt_name], "vibration_rule": C["vibration_rule"]}
    for c in ["leaky", "chosen", "alternative", "vibration_rule"]:
        p, r, _ = precision_recall_curve(test[TARGET], test[c])
        ls = ":" if c == "leaky" else "-"
        ap = obs_metrics(test[TARGET], test[c])["pr_auc"]
        ax.plot(r, p, ls, color=col[c], label=f"{lab[c]}  (PR-AUC {ap:.2f})")
    ax.axhline(test[TARGET].mean(), color="k", lw=0.6, ls="--", label=f"random ({test[TARGET].mean():.2f})")
    ax.set_xlabel("recall (share of at-risk observations flagged)"); ax.set_ylabel("precision")
    ax.set_title("Held-out test: Jan-Feb 2025")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    fig.tight_layout(); fig.savefig(FIG / "04_precision_recall_test.png"); plt.close(fig)


def tradeoff_fig(test, thr):
    s = sweep(test, "chosen")
    r = sweep(test, "vibration_rule")
    op = at_threshold(test, "chosen", thr)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(s.alerts_per_week, s.event_recall, color=C["logistic_regression"], label="chosen model")
    ax.plot(r.alerts_per_week, r.event_recall, color=C["vibration_rule"], label="vibration rule")
    ax.scatter([op["alerts_per_week"]], [op["event_recall"]], color="k", zorder=5)
    ax.annotate(f"operating point\n{op['event_recall']:.0%} failures caught\n{op['alerts_per_week']:.0f} alerts/week",
                (op["alerts_per_week"], op["event_recall"]), xytext=(12, -40), textcoords="offset points", fontsize=7.5,
                arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.set_xlabel("alerts per week across the 120-machine fleet"); ax.set_ylabel("share of failures caught (>=1 alert)")
    ax.set_title("Business trade-off on test: inspections vs failures caught")
    ax.legend(frameon=False, fontsize=8); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(FIG / "05_alerts_vs_failures_caught.png"); plt.close(fig)


def importance_fig(model, test, feats):
    pi = permutation_importance(model, test[feats], test[TARGET], scoring="average_precision",
                                n_repeats=5, random_state=0)
    imp = pd.Series(pi.importances_mean, index=feats).sort_values().tail(10)
    imp.to_csv(OUT / "permutation_importance_test.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.barh(imp.index, imp.values, color=C["logistic_regression"])
    ax.set_xlabel("drop in PR-AUC when shuffled (test)"); ax.set_title("What the model relies on (top 10)")
    fig.tight_layout(); fig.savefig(FIG / "06_feature_importance.png"); plt.close(fig)


if __name__ == "__main__":
    main()
