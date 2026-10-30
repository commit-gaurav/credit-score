"""Fairness auditing and disparate impact analysis for AltScore AI.

Implements algorithmic group fairness evaluation across demographic groups:
1. Gender (Female vs. Male)
2. Geography (Urban vs. Rural)

Rules enforced:
- Demographic attributes are strictly POST-SCORING metadata; they are NEVER
  passed to the predictive model during scoring.
- Uses a single documented approval threshold across all groups.
- Disparate impact ratio is computed strictly as:
      ratio = comparison_rate / reference_rate
  where the reference group is defined dynamically as the group with the
  higher approval rate, and the comparison group has the lower approval rate.
- Zero-rate handling: If the reference approval rate is 0.0, the disparate impact
  ratio is explicitly flagged as 'Undefined' rather than raising a ZeroDivisionError
  or inventing an arbitrary score.
- All numbers are derived dynamically from synthetic evaluation records.
"""

from typing import Any, Dict, Optional
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data.synthetic_data import get_feature_columns

# Standard documented approval threshold (repayment probability >= 70%)
DEFAULT_AUDIT_APPROVAL_THRESHOLD: float = 0.70


def calculate_fairness_metrics(
    model: LogisticRegression,
    evaluation_data: pd.DataFrame,
    threshold: float = DEFAULT_AUDIT_APPROVAL_THRESHOLD,
) -> Dict[str, Any]:
    """Calculate group approval rates and disparate impact ratios.

    Demographic columns are NEVER passed into model.predict_proba.
    The model predicts on the four financial features only; demographic labels
    are joined afterward strictly to partition the outcomes for auditing.

    Args:
        model: Fitted LogisticRegression scoring model.
        evaluation_data: Synthetic DataFrame containing features and demographic columns.
        threshold: Decision boundary probability for synthetic loan approval.

    Returns:
        Dict[str, Any]: Structured audit results containing group approval rates,
        identified reference/comparison groups, and disparate impact ratios.
    """
    feature_cols = get_feature_columns()

    # 1. Model inference: strictly feature-only inputs
    probabilities = model.predict_proba(evaluation_data[feature_cols])[:, 1]

    # Create a local copy with scored approval decisions
    audit_df = evaluation_data[["gender", "geography"]].copy()
    audit_df["repayment_probability"] = probabilities
    audit_df["approved"] = (probabilities >= threshold).astype(int)

    dimensions_result: Dict[str, Any] = {}
    audit_dimensions = [
        ("gender", "Gender"),
        ("geography", "Geography (Urban vs. Rural)"),
    ]

    for col, display_name in audit_dimensions:
        groups = sorted(audit_df[col].dropna().unique())
        group_stats: Dict[str, Dict[str, Any]] = {}

        for g in groups:
            sub = audit_df[audit_df[col] == g]
            total_n = len(sub)
            approved_n = int(sub["approved"].sum())
            rate = approved_n / total_n if total_n > 0 else 0.0
            group_stats[str(g)] = {
                "total": total_n,
                "approved": approved_n,
                "approval_rate": round(rate, 4),
            }

        # Determine reference (higher rate) and comparison (lower rate) groups
        # In case of exact tie, order deterministically
        sorted_groups = sorted(
            group_stats.keys(),
            key=lambda k: group_stats[k]["approval_rate"],
            reverse=True,
        )

        ref_group = sorted_groups[0]
        comp_group = sorted_groups[1] if len(sorted_groups) > 1 else sorted_groups[0]

        ref_rate = group_stats[ref_group]["approval_rate"]
        comp_rate = group_stats[comp_group]["approval_rate"]

        # Disparate impact ratio calculation with zero-rate handling
        di_ratio: Optional[float] = None
        ratio_display: str

        if ref_rate <= 0.0:
            # Handle zero reference rate explicitly
            ratio_display = "Undefined (Reference group approval rate is 0.0%)"
        else:
            di_ratio = round(comp_rate / ref_rate, 4)
            ratio_display = f"{di_ratio:.4f}"

        dimensions_result[col] = {
            "dimension_name": display_name,
            "groups": group_stats,
            "reference_group": ref_group,
            "reference_rate": ref_rate,
            "comparison_group": comp_group,
            "comparison_rate": comp_rate,
            "disparate_impact_ratio": di_ratio,
            "ratio_display": ratio_display,
        }

    return {
        "threshold": threshold,
        "total_records": len(evaluation_data),
        "dimensions": dimensions_result,
    }
