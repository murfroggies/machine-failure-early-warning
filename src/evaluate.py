"""Evaluation helpers: observation-level and failure-event-level metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import TARGET, TS


def obs_metrics(y: np.ndarray, s: np.ndarray) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "base_rate": float(np.mean(y)),
    }


def at_threshold(frame: pd.DataFrame, score: str, thr: float) -> dict:
    """Metrics when we alert on every observation with score >= thr.

    frame needs: TARGET, event_id (-1 for negatives), failure_est for events,
    TS, machine_id.
    """
    alert = frame[score].to_numpy() >= thr
    y = frame[TARGET].to_numpy()
    tp = int((alert & (y == 1)).sum())
    fp = int((alert & (y == 0)).sum())
    fn = int((~alert & (y == 1)).sum())

    pos = frame[y == 1].assign(alert=alert[y == 1])
    ev = pos.groupby("event_id").agg(
        caught=("alert", "max"), failure_est=("failure_est", "first")
    )
    first_alert = pos[pos["alert"]].groupby("event_id")[TS].min()
    lead = (ev.loc[first_alert.index, "failure_est"] - first_alert).dt.total_seconds() / 86400

    days = (frame[TS].max() - frame[TS].min()).total_seconds() / 86400
    weeks = max(days / 7, 1e-9)
    n_machines = frame["machine_id"].nunique()
    return {
        "threshold": float(thr),
        "obs_recall": tp / max(tp + fn, 1),
        "obs_precision": tp / max(tp + fp, 1),
        "event_recall": float(ev["caught"].mean()),
        "events_total": int(len(ev)),
        "events_caught": int(ev["caught"].sum()),
        "median_lead_days": float(lead.median()) if len(lead) else float("nan"),
        "alerts_per_week": (tp + fp) / weeks,
        "false_alerts_per_week": fp / weeks,
        "alerts_per_machine_per_month": (tp + fp) / weeks / n_machines * 4.345,
        "alert_rate": float(alert.mean()),
    }


def threshold_for_event_recall(frame: pd.DataFrame, score: str, target: float) -> float:
    """Highest threshold that still catches >= target share of failure events
    (fewest alerts that meet the recall goal)."""
    pos = frame[frame[TARGET] == 1]
    # an event is caught at threshold t if its max score >= t
    ev_max = pos.groupby("event_id")[score].max().sort_values(ascending=False).to_numpy()
    k = int(np.ceil(target * len(ev_max)))
    return float(ev_max[k - 1])


def sweep(frame: pd.DataFrame, score: str, n: int = 200) -> pd.DataFrame:
    qs = np.unique(np.quantile(frame[score], np.linspace(0.5, 0.999, n)))
    return pd.DataFrame([at_threshold(frame, score, t) for t in qs])


def machine_bootstrap(frame: pd.DataFrame, score: str, thr: float, n_boot: int = 500, seed: int = 0) -> dict:
    """95% intervals by resampling whole machines (observations of one
    machine are correlated, so row-level bootstrap would be too optimistic)."""
    rng = np.random.default_rng(seed)
    machines = frame["machine_id"].unique()
    groups = {m: g for m, g in frame.groupby("machine_id")}
    out = {"pr_auc": [], "event_recall": [], "obs_precision": [], "alerts_per_week": []}
    for b in range(n_boot):
        pick = rng.choice(machines, size=len(machines), replace=True)
        parts = []
        for i, m in enumerate(pick):
            g = groups[m].copy()
            g["machine_id"] = f"{m}_{i}"
            g["event_id"] = np.where(g["event_id"] >= 0, g["event_id"] * 1000 + i, -1)
            parts.append(g)
        s = pd.concat(parts)
        if s[TARGET].sum() == 0:
            continue
        out["pr_auc"].append(average_precision_score(s[TARGET], s[score]))
        m = at_threshold(s, score, thr)
        for k in ("event_recall", "obs_precision", "alerts_per_week"):
            out[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}
