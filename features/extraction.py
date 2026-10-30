"""Financial feature extraction from synthetic transaction histories.

Implements the four core alternative credit signals defined in PLAN.md:
1. Income stability (0.0 to 1.0): Consistency of monthly income across observation periods.
2. Bill-payment consistency (0.0 to 1.0): Ratio of recorded bills paid on time.
3. Payment regularity (0.0 to 1.0): Fraction of observation months with incoming payments.
4. Financial liquidity buffer (0.0 to 6.0 months): Liquid balance relative to essential monthly outflows.

Rules enforced:
- Explicit missing-data handling (missing transactions never silently treated as missed payments).
- Demographic attributes are strictly excluded from all calculations.
- Liquidity buffer is capped at exactly 6.0 months.
"""

import math
from typing import Any, Dict, List


def calculate_income_stability(transactions: List[Dict[str, Any]]) -> float:
    """Calculate normalized monthly income stability (0.0 to 1.0).

    Formula:
        stability = max(0.0, 1.0 - (standard_deviation / mean_monthly_income))
        Higher value indicates more predictable, consistent monthly cash inflow.

    Missing-data handling:
        - If no income transactions exist: returns 0.0 (no observable income).
        - If only 1 month exists: standard deviation is 0.0 (perfect consistency across
          available observation), returns 1.0.

    Args:
        transactions: List of transaction records.

    Returns:
        float: Income stability score rounded to 4 decimals, bounded in [0.0, 1.0].
    """
    # Filter for incoming income transactions
    income_by_month: Dict[str, float] = {}
    for tx in transactions:
        if tx.get("category") == "Income" or tx.get("type") == "Credit":
            month_key = str(tx.get("date", ""))[:7]  # YYYY-MM
            income_by_month[month_key] = income_by_month.get(month_key, 0.0) + float(tx.get("amount", 0.0))

    if not income_by_month:
        return 0.0

    monthly_totals = list(income_by_month.values())
    count = len(monthly_totals)
    if count == 1:
        # Only one month observed; no cross-month variance observed
        return 1.0

    mean_income = sum(monthly_totals) / count
    if mean_income <= 0.0:
        return 0.0

    # Sample standard deviation (ddof=1)
    variance = sum((x - mean_income) ** 2 for x in monthly_totals) / (count - 1)
    std_dev = math.sqrt(variance)

    # Coefficient of variation = std_dev / mean
    cv = std_dev / mean_income

    # Normalized score: 1.0 means perfectly flat income; drops towards 0 as variance rises
    stability = max(0.0, min(1.0, 1.0 - cv))
    return round(stability, 4)


def calculate_bill_consistency(transactions: List[Dict[str, Any]]) -> float:
    """Calculate the ratio of recorded utility bills paid on time (0.0 to 1.0).

    Formula:
        bill_consistency = count(bills with status 'Paid on time') / count(all bills)

    Missing-data handling:
        CRITICAL RULE: Missing transactions must NOT silently count as missed payments.
        If an applicant has 0 recorded utility bills, consistency is NOT set to 0.0.
        Instead, we return 1.0 (no adverse late/unpaid events recorded) and note that
        absence of bills does not imply delinquency.

    Args:
        transactions: List of transaction records.

    Returns:
        float: Bill payment consistency ratio bounded in [0.0, 1.0].
    """
    bills = [tx for tx in transactions if tx.get("category") == "Utility Bill"]

    if not bills:
        # Explicit missing-data handling: no bills recorded != missed payments
        return 1.0

    on_time_count = sum(1 for b in bills if b.get("status") == "Paid on time")
    consistency = on_time_count / len(bills)
    return round(max(0.0, min(1.0, consistency)), 4)


def calculate_payment_regularity(transactions: List[Dict[str, Any]]) -> float:
    """Calculate the fraction of observation periods with a recorded incoming payment (0.0 to 1.0).

    Formula:
        regularity = count(months with >=1 income payment) / count(total observation months)

    Missing-data handling:
        If no transactions exist across any category, observation span is 0 months; returns 0.0.

    Args:
        transactions: List of transaction records.

    Returns:
        float: Payment regularity ratio bounded in [0.0, 1.0].
    """
    if not transactions:
        return 0.0

    # Total observed calendar months in the transaction record
    all_months = {str(tx.get("date", ""))[:7] for tx in transactions if tx.get("date")}
    if not all_months:
        return 0.0

    # Months with recorded positive income
    income_months = {
        str(tx.get("date", ""))[:7]
        for tx in transactions
        if (tx.get("category") == "Income" or tx.get("type") == "Credit") and float(tx.get("amount", 0.0)) > 0.0
    }

    regularity = len(income_months) / len(all_months)
    return round(max(0.0, min(1.0, regularity)), 4)


def calculate_liquidity_buffer(
    transactions: List[Dict[str, Any]],
    liquid_balance: float = 0.0,
) -> float:
    """Calculate liquid savings buffer in months of essential outflow, capped at 6.0 months.

    Formula:
        monthly_essential_outflow = sum(essential expenses) / count(observation months)
        liquidity_buffer = min(6.0, liquid_balance / monthly_essential_outflow)

    Missing-data handling:
        - If average monthly essential outflow is 0.0 or unrecorded:
          - If liquid_balance > 0, buffer is effectively high; returns capped 6.0 months.
          - If liquid_balance <= 0, returns 0.0.
        - Negative liquid balances are floored at 0.0.
        - Result is strictly capped at 6.0 months per PLAN.md specification.

    Args:
        transactions: List of transaction records.
        liquid_balance: Current available cash / savings in bank or wallet (in INR).

    Returns:
        float: Liquidity buffer in months, bounded in [0.0, 6.0].
    """
    if liquid_balance <= 0.0:
        return 0.0

    # Essential outflows (e.g. food, fuel, essential inventory restocking)
    essential_txs = [
        tx for tx in transactions
        if tx.get("category") == "Essential Outflow" or (tx.get("type") == "Debit" and tx.get("category") != "Utility Bill")
    ]

    all_months = {str(tx.get("date", ""))[:7] for tx in transactions if tx.get("date")}
    month_count = len(all_months) if all_months else 1

    total_essential_outflow = sum(float(tx.get("amount", 0.0)) for tx in essential_txs)
    avg_monthly_outflow = total_essential_outflow / month_count

    if avg_monthly_outflow <= 0.0:
        # No essential outflows recorded; liquid balance safely covers > 6 months
        return 6.0

    buffer_months = liquid_balance / avg_monthly_outflow
    # Cap strictly at 6.0 months
    capped_buffer = min(6.0, max(0.0, buffer_months))
    return round(capped_buffer, 2)


def extract_features(
    transactions: List[Dict[str, Any]],
    liquid_balance: float = 0.0,
) -> Dict[str, float]:
    """Extract all four alternative credit features from transaction records.

    Note: Demographic attributes (gender, urban/rural) are deliberately NOT accepted
    or calculated here to prevent demographic bias in model features.

    Args:
        transactions: List of synthetic transaction records.
        liquid_balance: Current liquid balance for liquidity buffer calculation.

    Returns:
        Dict[str, float]: Extracted feature dictionary:
            - income_stability: normalized monthly stability [0.0 - 1.0]
            - bill_payment_consistency: fraction of bills paid on time [0.0 - 1.0]
            - payment_regularity: fraction of months with income [0.0 - 1.0]
            - financial_liquidity_buffer: savings buffer in months [0.0 - 6.0]
    """
    return {
        "income_stability": calculate_income_stability(transactions),
        "bill_payment_consistency": calculate_bill_consistency(transactions),
        "payment_regularity": calculate_payment_regularity(transactions),
        "financial_liquidity_buffer": calculate_liquidity_buffer(transactions, liquid_balance),
    }

