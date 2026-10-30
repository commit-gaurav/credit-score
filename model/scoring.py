"""Model training, scoring, and evaluation for AltScore AI.

Implements baseline logistic regression training on synthetic alternative
credit signals.

Rules enforced:
- Strictly reproducible train/test split with fixed random seed.
- Fits only on the four behavioral financial features.
- Demographic attributes and target labels are never exposed to model inputs.
- Evaluation metrics are explicitly computed on the held-out test split
  and labeled as synthetic evaluation results.
"""

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data.synthetic_data import get_feature_columns, get_target_column

DEFAULT_MODEL_SEED: int = 42
DEFAULT_TEST_SIZE: float = 0.25


def train_model(
    data: pd.DataFrame,
    seed: int = DEFAULT_MODEL_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Train a baseline logistic regression model and compute held-out metrics.

    Separates features X and target y, performs a stratified held-out split,
    fits a standard logistic regression classifier, and evaluates performance
    on the unseen test split.

    Args:
        data: Full synthetic dataset containing features and target column.
        seed: Random seed for train/test split and model fitting.
        test_size: Fraction of rows reserved for held-out evaluation (default 0.25).

    Returns:
        Tuple[LogisticRegression, Dict[str, Any]]:
            - Fitted LogisticRegression model instance
            - Evaluation metrics dictionary containing ROC-AUC, accuracy, precision,
              recall, F1-score, sample sizes, and model coefficients.
    """
    feature_cols = get_feature_columns()
    target_col = get_target_column()

    # Feature matrix X contains strictly the 4 behavioral features (no demographics)
    X = data[feature_cols]
    y = data[target_col]

    # Stratified train/test split ensures identical target proportions in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Initialize and fit explainable linear classifier
    model = LogisticRegression(random_state=seed, max_iter=1000)
    model.fit(X_train, y_train)

    # Held-out predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics on held-out test split
    metrics: Dict[str, Any] = {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_names": feature_cols,
        "coefficients": {
            col: round(float(coef), 4)
            for col, coef in zip(feature_cols, model.coef_[0])
        },
        "intercept": round(float(model.intercept_[0]), 4),
    }

    return model, metrics

