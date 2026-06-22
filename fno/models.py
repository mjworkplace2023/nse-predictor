"""ML models — XGBoost, ensemble, optional LSTM."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from fno.config import RANDOM_STATE, TRAIN_TEST_SPLIT
from fno.features import FEATURE_COLUMNS
from fno.labels import LABEL_MAP

logger = logging.getLogger(__name__)


class FnoModelBundle:
    """Trained models + scaler for inference."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.ensemble = None
        self.xgb_model = None
        self.train_accuracy = 0.0
        self.feature_columns: List[str] = []

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.feature_columns if c in X.columns]
        x = self.scaler.transform(X[cols])
        if self.ensemble is not None:
            return self.ensemble.predict_proba(x)[0]
        if self.xgb_model is not None:
            return self.xgb_model.predict_proba(x)[0]
        return np.array([0.33, 0.34, 0.33])

    def predict_label(self, X: pd.DataFrame) -> Tuple[str, float]:
        proba = self.predict_proba(X)
        idx = int(np.argmax(proba))
        return LABEL_MAP[idx], float(proba[idx])


def time_series_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — never shuffle financial data."""
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_models(labeled_df: pd.DataFrame) -> FnoModelBundle:
    """Train XGBoost + voting ensemble on time-series split data."""
    bundle = FnoModelBundle()
    cols = [c for c in FEATURE_COLUMNS if c in labeled_df.columns]
    bundle.feature_columns = cols

    train_df, test_df = time_series_split(labeled_df)
    if len(train_df) < 50 or len(test_df) < 10:
        logger.warning("Insufficient labeled rows for training (%d)", len(labeled_df))
        return bundle

    X_train = train_df[cols]
    y_train = train_df["label"].astype(int)
    X_test = test_df[cols]
    y_test = test_df["label"].astype(int)

    bundle.scaler.fit(X_train)
    X_train_s = bundle.scaler.transform(X_train)
    X_test_s = bundle.scaler.transform(X_test)

    estimators = []

    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        )
        xgb.fit(X_train_s, y_train)
        bundle.xgb_model = xgb
        estimators.append(("xgb", xgb))
    except Exception as exc:
        logger.warning("XGBoost unavailable: %s", exc)

    try:
        from lightgbm import LGBMClassifier

        lgbm = LGBMClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        lgbm.fit(X_train_s, y_train)
        estimators.append(("lgbm", lgbm))
    except Exception as exc:
        logger.warning("LightGBM unavailable: %s", exc)

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE)
    rf.fit(X_train_s, y_train)
    estimators.append(("rf", rf))

    lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(X_train_s, y_train)
    estimators.append(("lr", lr))

    if estimators:
        bundle.ensemble = VotingClassifier(estimators=estimators, voting="soft")
        bundle.ensemble.fit(X_train_s, y_train)
        preds = bundle.ensemble.predict(X_test_s)
        bundle.train_accuracy = float(accuracy_score(y_test, preds))

    return bundle


def lstm_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False
