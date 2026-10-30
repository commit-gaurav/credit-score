"""AltScore AI — Streamlit Application Dashboard.

Slice 3: Calculates and displays the four alternative financial features
extracted from the selected persona's transaction history.
"""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from data.personas import get_personas
from features.extraction import extract_features


def render_header() -> None:
    """Render the application title and brief purpose statement."""
    st.title("AltScore AI")
    st.subheader("Alternative Credit Scoring & Scenario Exploration")
    st.markdown(
        "AltScore AI is an exploratory prototype designed to estimate repayment risk "
        "and explore financial scenarios for individuals with thin or non-traditional "
        "credit files (e.g., gig workers, small shopkeepers, and first-time borrowers) "
        "using alternative behavioral signals and explainable machine learning."
    )


def render_disclaimer() -> None:
    """Render the synthetic research prototype disclaimer.

    Ensures judges and reviewers immediately recognize that this is a research
    prototype and not an actual credit underwriting engine.
    """
    st.warning(
        "**RESEARCH PROTOTYPE DISCLAIMER:**\n\n"
        "AltScore AI is a synthetic research and educational prototype developed for "
        "hackathon evaluation. All transaction records, features, and model outputs "
        "are purely synthetic demonstrations. This application does NOT produce real "
        "credit decisions, determine real loan eligibility, or reflect real-world creditworthiness."
    )


def render_persona_selector(personas: Dict[str, Dict[str, Any]]) -> str:
    """Render the persona selection dropdown and return the selected persona key.

    Args:
        personas: Dictionary of available applicant profiles.

    Returns:
        str: Selected persona identifier key.
    """
    st.markdown("### 1. Select Synthetic Applicant Profile")
    persona_keys = list(personas.keys())

    # Format options with applicant name and role for clarity
    selected_key = st.selectbox(
        label="Choose an applicant to inspect their financial transaction history:",
        options=persona_keys,
        format_func=lambda key: f"{personas[key]['name']} — {personas[key]['role']}",
    )
    return selected_key


def render_persona_profile(persona: Dict[str, Any]) -> None:
    """Display profile details, liquid balance, and demographic audit metadata.

    Args:
        persona: The selected persona dictionary containing profile and metadata.
    """
    st.markdown(f"#### Profile: **{persona['name']}**")
    st.caption(f"Role: {persona['role']}")
    st.write(persona["summary"])

    # Highlight metrics and metadata in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Current Liquid Balance",
            value=f"₹{persona['liquid_balance']:,.2f}",
        )
    with col2:
        # Demographics are displayed as audit metadata; not model features
        st.metric(
            label="Audit Group: Gender",
            value=persona["demographics"]["gender"],
        )
    with col3:
        st.metric(
            label="Audit Group: Geography",
            value=persona["demographics"]["geography"],
        )

    st.caption(
        "Note: Demographic attributes (Gender, Geography) are tracked solely for "
        "disparate impact audit analysis (Slice 9) and are **never** used as predictive model features."
    )


def render_transaction_table(transactions: List[Dict[str, Any]]) -> None:
    """Display the persona's synthetic transaction history in a clean table.

    Args:
        transactions: List of transaction records for the selected persona.
    """
    st.markdown("### 2. Synthetic Transaction Records")

    # Convert records into a pandas DataFrame for structured display
    df = pd.DataFrame(transactions)

    # Rename columns to clear presentation headers
    df_display = df.rename(
        columns={
            "date": "Date",
            "description": "Description",
            "category": "Category",
            "type": "Type",
            "amount": "Amount (₹)",
            "status": "Status",
        }
    )

    # Format numeric amounts with comma separation and 2 decimals
    df_display["Amount (₹)"] = df_display["Amount (₹)"].apply(lambda amt: f"₹{amt:,.2f}")

    st.dataframe(df_display, use_container_width=True, hide_index=True)
    st.caption(f"Showing all {len(df_display)} recorded synthetic transactions.")


def render_extracted_features(features: Dict[str, float]) -> None:
    """Display the four extracted alternative credit features and their documentation.

    Args:
        features: Dictionary of calculated feature values.
    """
    st.markdown("### 3. Extracted Financial Features")
    st.write(
        "The following behavioral features are calculated directly from transaction history "
        "and liquid cash reserves. Demographic fields are completely excluded."
    )

    # Display 4 metrics in horizontal columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Income Stability",
            value=f"{features['income_stability']:.2f}",
            help="Range: 0.00–1.00. 1.00 = completely consistent monthly income across months.",
        )
    with col2:
        st.metric(
            label="Bill-Payment Consistency",
            value=f"{features['bill_payment_consistency']:.2f}",
            help="Range: 0.00–1.00. Fraction of recorded utility bills paid on time.",
        )
    with col3:
        st.metric(
            label="Payment Regularity",
            value=f"{features['payment_regularity']:.2f}",
            help="Range: 0.00–1.00. Fraction of observation months with >= 1 incoming payment.",
        )
    with col4:
        st.metric(
            label="Liquidity Buffer",
            value=f"{features['financial_liquidity_buffer']:.2f} mos",
            help="Range: 0.00–6.00 months. Liquid balance / average monthly essential outflow.",
        )

    # Document formulas and missing data policies in an expander
    with st.expander("📖 Feature Definitions & Missing-Data Documentation"):
        st.markdown(
            """
            | Feature | Unit & Range | Formula | Explicit Missing-Data Policy |
            | :--- | :--- | :--- | :--- |
            | **Income Stability** | Score (0.00–1.00) | $\\max(0, 1 - \\sigma / \\mu)$ across monthly incomes | If no income is recorded, returns `0.00`. If only 1 month is available, variation is 0, returns `1.00`. |
            | **Bill-Payment Consistency** | Ratio (0.00–1.00) | $\\text{On-time bills} / \\text{Total bills}$ | **Zero bills recorded does NOT count as missed payments.** Default is neutral `1.00` (no adverse events recorded). |
            | **Payment Regularity** | Ratio (0.00–1.00) | $\\text{Months with income} / \\text{Total observed months}$ | If zero transactions exist in account history, returns `0.00`. |
            | **Financial Liquidity Buffer** | Months (0.00–6.00) | $\\min(6.0, \\text{Liquid Balance} / \\text{Monthly Essential Outflow})$ | If zero essential outflows exist, capped at `6.00` months. Negative balance floored at `0.00`. Strictly capped at `6.00` months. |
            """
        )


def main() -> None:
    """Build the dashboard and render current slice components."""
    # st.set_page_config must be called as the first Streamlit command
    st.set_page_config(
        page_title="AltScore AI",
        page_icon="💳",
        layout="wide",
    )

    render_header()
    render_disclaimer()

    st.divider()

    # Load in-memory synthetic profiles
    personas = get_personas()

    # Persona selection and profile display
    selected_key = render_persona_selector(personas)
    selected_persona = personas[selected_key]

    render_persona_profile(selected_persona)

    st.divider()

    # Transaction records table
    render_transaction_table(selected_persona["transactions"])

    st.divider()

    # Feature extraction and display
    features = extract_features(
        transactions=selected_persona["transactions"],
        liquid_balance=selected_persona["liquid_balance"],
    )
    render_extracted_features(features)

    st.divider()
    st.info(
        "**Slice 3 Active:** Four financial features extracted from transactions. "
        "Synthetic training data generation (Slice 4) and baseline scoring model (Slice 5) "
        "will be implemented in subsequent slices."
    )


if __name__ == "__main__":
    main()
