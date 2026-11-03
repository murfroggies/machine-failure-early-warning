"""Feature engineering.

Rule: every feature for an observation uses only that observation and the
same machine's EARLIER observations, because the model scores each record at
the moment it is recorded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TS

SENSORS = [
    "vibration_mm_s",
    "bearing_temp_c",
    "motor_current_a",
    "discharge_pressure_kpa",
]
CONTEXT = ["load_pct", "ambient_temp_c"]
CATEGORICAL = ["site", "machine_class"]

# Excluded after the leakage analysis (see docs/ASSUMPTIONS.md, section 2)
LEAKY = ["corrective_orders_14d", "inspection_interval_days"]
# Kept, but timing to be confirmed with the data owner
OPERATIONAL = ["unplanned_stops_14d"]

BASELINE_WINDOW = 8  # previous observations (~40 days) for a machine's "normal"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machine_id", TS]).copy()
    g = df.groupby("machine_id", sort=False)

    for c in SENSORS:
        # Machine's own recent baseline from PAST observations only.
        past = g[c].shift(1)
        base = past.groupby(df["machine_id"]).transform(
            lambda s: s.rolling(BASELINE_WINDOW, min_periods=3).median()
        )
        df[f"{c}_vs_baseline"] = df[c] - base
        # Short-term trend: change since previous observation.
        df[f"{c}_delta1"] = df[c] - past

    # 3-observation vibration slope (per day), past + current only.
    t_days = (df[TS] - df[TS].min()).dt.total_seconds() / 86400
    v_lag2 = g["vibration_mm_s"].shift(2)
    t_lag2 = t_days.groupby(df["machine_id"]).shift(2)
    df["vibration_slope_3obs"] = (df["vibration_mm_s"] - v_lag2) / (t_days - t_lag2)

    df["bearing_rise_over_ambient"] = df["bearing_temp_c"] - df["ambient_temp_c"]
    df["current_per_load"] = df["motor_current_a"] / df["load_pct"]
    df["days_since_prev_obs"] = t_days.groupby(df["machine_id"]).diff()

    df["vibration_missing"] = df["vibration_mm_s"].isna().astype(int)
    df["bearing_temp_missing"] = df["bearing_temp_c"].isna().astype(int)

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df.sort_index()


def feature_sets() -> dict[str, list[str]]:
    eng = (
        [f"{c}_vs_baseline" for c in SENSORS]
        + [f"{c}_delta1" for c in SENSORS]
        + [
            "vibration_slope_3obs",
            "bearing_rise_over_ambient",
            "current_per_load",
            "days_since_prev_obs",
            "vibration_missing",
            "bearing_temp_missing",
        ]
    )
    base = SENSORS + CONTEXT + CATEGORICAL
    return {
        # what we recommend
        "sensors_plus_history": base + eng + OPERATIONAL,
        # sensitivity: drop the one field whose timing is unconfirmed
        "sensors_only": base + eng,
        # for illustration ONLY: what you get if the leaky fields are used
        "leaky_all_fields": base + eng + OPERATIONAL + LEAKY,
    }
