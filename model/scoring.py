"""Model training, scoring, and evaluation for AltScore AI.

Implements baseline logistic regression training on synthetic alternative
credit signals, probability estimation, illustrative score mapping, and
risk categorization.

Rules enforced:
- Strictly reproducible train/test split with fixed random seed.
- Fits only on the four behavioral financial features.
- Demographic attributes and target labels are never exposed to model inputs.
- Evaluation metrics are explicitly computed on the held-out test split
  and labeled as synthetic evaluation results.
- Score mapping follows the exact formula: 300 + 600 * probability.
- All numbers are dynamically computed from data and model; no hardcoding.
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

# Documented Risk Category Thresholds (Repayment Probability)
LOW_RISK_THRESHOLD: float = 0.85    # Equivalent to illustrative score >= 810
MEDIUM_RISK_THRESHOLD: float = 0.70  # Equivalent to illustrative score 720 - 809


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


def predict_repayment_probability(
    model: LogisticRegression,
    features: Dict[str, float],
) -> float:
    """Calculate the model's estimated probability for simulated loan repayment.

    Uses the actual fitted logistic regression model. Features are ordered
    strictly according to the feature column definitions.

    Args:
        model: Fitted LogisticRegression classifier.
        features: Dictionary containing the four behavioral features.

    Returns:
        float: Estimated repayment probability bounded in [0.0, 1.0], rounded to 4 decimals.
    """
    feature_cols = get_feature_columns()
    input_df = pd.DataFrame([[features[col] for col in feature_cols]], columns=feature_cols)
    prob = float(model.predict_proba(input_df)[0, 1])
    return round(float(prob), 4)


def probability_to_score(probability: float) -> int:
    """Map repayment probability to an illustrative 300-900 credit score.

    Exact formula from PLAN.md:
        illustrative_score = round(300 + 600 * probability)
        Clamped to [300, 900].

    Note:
        This is an illustrative linear transformation of the model's estimated
        probability, NOT an established credit bureau score (such as CIBIL or FICO).

    Args:
        probability: Repayment probability in [0.0, 1.0].

    Returns:
        int: Bounded illustrative score between 300 and 900.
    """
    raw_score = 300.0 + (600.0 * probability)
    clamped_score = max(300, min(900, int(round(raw_score))))
    return clamped_score


def categorize_risk(probability: float) -> str:
    """Map repayment probability to documented demo risk categories.

    Thresholds:
        - Low Risk: Probability >= 0.85 (Illustrative Score 810-900)
        - Medium Risk: 0.70 <= Probability < 0.85 (Illustrative Score 720-809)
        - High Risk: Probability < 0.70 (Illustrative Score 300-719)

    Args:
        probability: Repayment probability in [0.0, 1.0].

    Returns:
        str: Risk category label ('Low Risk', 'Medium Risk', or 'High Risk').
    """
    if probability >= LOW_RISK_THRESHOLD:
        return "Low Risk"
    elif probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium Risk"
    else:
        return "High Risk"
