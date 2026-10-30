"""Reproducible synthetic dataset generator for AltScore AI.

Generates synthetic applicant records with:
1. Four behavioral financial features:
   - income_stability [0.0 - 1.0]
   - bill_payment_consistency [0.0 - 1.0]
   - payment_regularity [0.0 - 1.0]
   - financial_liquidity_buffer [0.0 - 6.0 months]
2. Demographic attributes for fairness auditing only:
   - gender: 'Female' or 'Male'
   - geography: 'Urban' or 'Rural'
3. Simulated binary repayment target:
   - repaid: 1 (repaid successfully) or 0 (defaulted / missed repayment)

CRITICAL RULE ENFORCEMENT:
- Demographic attributes are kept strictly separate from model input features.
- Target generation uses a documented latent logistic model with stochastic noise,
  preventing trivial separability while capturing realistic credit risk relationships.
- Uses a fixed random seed (default: 42) for deterministic, reproducible results.
"""

from typing import List
import numpy as np
import pandas as pd

DEFAULT_SEED: int = 42
DEFAULT_ROW_COUNT: int = 1000

# Explicit separation of column roles
FEATURE_COLUMNS: List[str] = [
    "income_stability",
    "bill_payment_consistency",
    "payment_regularity",
    "financial_liquidity_buffer",
]

DEMOGRAPHIC_COLUMNS: List[str] = [
    "gender",
    "geography",
]

TARGET_COLUMN: str = "repaid"


def get_feature_columns() -> List[str]:
    """Return the list of financial feature column names used for model training."""
    return list(FEATURE_COLUMNS)


def get_demographic_columns() -> List[str]:
    """Return the list of demographic metadata column names reserved for fairness auditing."""
    return list(DEMOGRAPHIC_COLUMNS)


def get_target_column() -> str:
    """Return the target column name for repayment outcome."""
    return TARGET_COLUMN


def generate_synthetic_dataset(
    seed: int = DEFAULT_SEED,
    row_count: int = DEFAULT_ROW_COUNT,
) -> pd.DataFrame:
    """Generate a reproducible synthetic dataset for training and auditing.

    Target generation formula:
        logit = beta_0 + sum(beta_i * feature_i) + epsilon
        where epsilon ~ Normal(0, 0.6) introduces realistic unobserved life shocks
        (e.g., medical emergencies, unexpected expenses).
        repayment_probability = 1 / (1 + exp(-logit))
        repaid ~ Bernoulli(repayment_probability)

    Demographics are generated independently as audit metadata and have NO direct
    causal weight in the repayment target equation, preventing direct demographic bias.

    Args:
        seed: Random seed for exact reproducibility.
        row_count: Number of synthetic applicant records to generate.

    Returns:
        pd.DataFrame: DataFrame containing features, demographics, and target label.
    """
    rng = np.random.default_rng(seed)

    # 1. Demographic metadata (evaluated in Slice 9 audit, never input to model)
    gender = rng.choice(["Female", "Male"], size=row_count, p=[0.48, 0.52])
    geography = rng.choice(["Rural", "Urban"], size=row_count, p=[0.45, 0.55])

    # 2. Four financial behavioral features
    # Modeled with beta and exponential distributions matching real-world credit ranges
    income_stability = np.clip(rng.beta(a=7.0, b=2.5, size=row_count), 0.0, 1.0)
    bill_consistency = np.clip(rng.beta(a=8.0, b=2.5, size=row_count), 0.0, 1.0)
    payment_regularity = np.clip(rng.beta(a=8.0, b=2.0, size=row_count), 0.0, 1.0)
    liquidity_buffer = np.clip(rng.exponential(scale=1.8, size=row_count), 0.0, 6.0)

    # 3. Simulated repayment target (documented latent logistic model)
    # Weights reflect intuitive financial principles:
    # - On-time bill payment and liquidity buffers strongly support repayment
    # - Income stability and regular incoming payments provide baseline support
    # - Normal noise represents unpredictable life emergencies
    # Notice: 'gender' and 'geography' are completely excluded from logit calculation
    logit = (
        -3.2
        + 1.8 * income_stability
        + 2.6 * bill_consistency
        + 1.5 * payment_regularity
        + 1.5 * (liquidity_buffer / 6.0)
        + rng.normal(loc=0.0, scale=0.6, size=row_count)
    )

    # Logistic sigmoid function converts logit to continuous probability [0, 1]
    prob_repayment = 1.0 / (1.0 + np.exp(-logit))

    # Bernoulli trial creates binary outcome (1 = Repaid on time, 0 = Defaulted)
    repaid = (rng.uniform(0.0, 1.0, size=row_count) < prob_repayment).astype(int)

    # Construct clean, structured DataFrame
    df = pd.DataFrame(
        {
            "income_stability": np.round(income_stability, 4),
            "bill_payment_consistency": np.round(bill_consistency, 4),
            "payment_regularity": np.round(payment_regularity, 4),
            "financial_liquidity_buffer": np.round(liquidity_buffer, 2),
            "gender": gender,
            "geography": geography,
            "repaid": repaid,
        }
    )

    return df

