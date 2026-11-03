"""Candidate models. Hyperparameters are fixed, sensible defaults: with ~340
failure events, tuning many knobs would mostly fit noise."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL

SEED = 42


def logistic(features: list[str]) -> Pipeline:
    num = [f for f in features if f not in CATEGORICAL]
    cat = [f for f in features if f in CATEGORICAL]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                ("sc", StandardScaler()),
            ]), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )
    return Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)),
    ])


def gradient_boosting(features: list[str]) -> HistGradientBoostingClassifier:
    # Handles missing values natively; categories via pandas 'category' dtype.
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        class_weight="balanced",
        categorical_features="from_dtype",
        early_stopping=False,
        random_state=SEED,
    )


MODELS = {"logistic_regression": logistic, "gradient_boosting": gradient_boosting}
