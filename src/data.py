"""Loading and cleaning of machine_observations.csv.

Every cleaning step is logged so the README / report can quote the counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_TO_KPA = 6.894757
# South-site discharge pressure switches to roughly 1/6.9 of its normal level
# from this point on (all 24 South machines at once, other sites unaffected).
# Multiplying by 6.89 restores the original distribution exactly, so we treat
# it as a psi feed being written into a kPa column. See docs/ASSUMPTIONS.md.
PRESSURE_UNIT_THRESHOLD_KPA = 200.0

TARGET = "fails_within_7d"
TS = "observation_timestamp"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TS])
    return df.sort_values(["machine_id", TS, "record_id"]).reset_index(drop=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log: dict = {"rows_raw": int(len(df))}
    df = df.copy()

    # 1. Exact duplicate observations (same machine, time and readings,
    #    different record_id). Keep the first.
    value_cols = [c for c in df.columns if c != "record_id"]
    dup = df.duplicated(subset=value_cols, keep="first")
    log["exact_duplicates_dropped"] = int(dup.sum())
    df = df[~dup].copy()

    # 2. Sentinel value -999 in vibration -> missing.
    sentinel = df["vibration_mm_s"] == -999
    log["vibration_sentinel_to_nan"] = int(sentinel.sum())
    df.loc[sentinel, "vibration_mm_s"] = np.nan

    # 3. Pressure unit fix (psi written as kPa).
    psi = df["discharge_pressure_kpa"] < PRESSURE_UNIT_THRESHOLD_KPA
    log["pressure_psi_rows_converted"] = int(psi.sum())
    log["pressure_psi_sites"] = sorted(df.loc[psi, "site"].unique().tolist())
    log["pressure_psi_from"] = str(df.loc[psi, TS].min())
    df.loc[psi, "discharge_pressure_kpa"] *= PSI_TO_KPA

    # 4. machine_class 'unknown' -> the machine's own known class
    #    (class is a fixed attribute; every affected machine has exactly one
    #    other class on all its other rows).
    known = (
        df[df["machine_class"] != "unknown"]
        .groupby("machine_id")["machine_class"]
        .agg(lambda s: s.mode().iat[0])
    )
    unk = df["machine_class"] == "unknown"
    log["machine_class_unknown_filled"] = int(unk.sum())
    df.loc[unk, "machine_class"] = df.loc[unk, "machine_id"].map(known)

    log["missing_after_clean"] = {
        k: int(v) for k, v in df.isna().sum().items() if v > 0
    }
    log["rows_clean"] = int(len(df))
    return df.reset_index(drop=True), log


def reconstruct_failures(df: pd.DataFrame) -> pd.DataFrame:
    """Infer failure events from the label.

    A run of consecutive positive observations for one machine is one failure
    episode. The failure time F must satisfy
        last_positive < F <= first_positive + 7d
        F > previous_negative + 7d   and   F <= next_negative
    We keep the feasible window [lo, hi] and use its midpoint as the
    estimate. Used only for analysis and event-level evaluation, never as a
    model input.
    """
    rows = []
    for mid, g in df.groupby("machine_id", sort=False):
        t = g[TS].to_numpy()
        y = g[TARGET].to_numpy()
        i = 0
        while i < len(y):
            if y[i] != 1:
                i += 1
                continue
            j = i
            while j + 1 < len(y) and y[j + 1] == 1:
                j += 1
            lo = t[j]
            if i > 0:
                lo = max(lo, t[i - 1] + np.timedelta64(7, "D"))
            hi = t[i] + np.timedelta64(7, "D")
            if j + 1 < len(y):
                hi = min(hi, t[j + 1])
            rows.append(
                dict(machine_id=mid, first_positive=t[i], lo=lo, hi=hi, n_positive_obs=j - i + 1)
            )
            i = j + 1
    ev = pd.DataFrame(rows)
    ev["failure_est"] = ev["lo"] + (ev["hi"] - ev["lo"]) / 2
    ev["event_id"] = np.arange(len(ev))
    return ev


def attach_event_info(df: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Signed days from each observation to the nearest estimated failure
    (positive = failure is in the future) and the event id for positives."""
    df = df.copy()
    df["days_to_nearest_failure"] = np.nan
    df["event_id"] = -1
    for mid, e in ev.groupby("machine_id"):
        idx = df.index[df["machine_id"] == mid]
        t = df.loc[idx, TS].to_numpy()
        f = e["failure_est"].to_numpy()
        d = (f[None, :] - t[:, None]) / np.timedelta64(1, "D")
        k = np.abs(d).argmin(axis=1)
        df.loc[idx, "days_to_nearest_failure"] = d[np.arange(len(idx)), k]
        # event a positive belongs to = next failure ahead
        ahead = np.where(d > 0, d, np.inf)
        k2 = ahead.argmin(axis=1)
        ok = np.isfinite(ahead.min(axis=1))
        eid = np.where(ok, e["event_id"].to_numpy()[k2], -1)
        df.loc[idx, "event_id"] = np.where(df.loc[idx, TARGET] == 1, eid, -1)
    return df
