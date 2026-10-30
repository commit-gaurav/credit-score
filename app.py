"""AltScore AI — Streamlit Application Dashboard.

Slice 2: Displays synthetic applicant personas and their transaction histories.
Demographic attributes are presented strictly as profile/audit metadata
and are separated from any predictive modeling features.
"""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from data.personas import get_personas


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
    st.info(
        "**Slice 2 Active:** Synthetic personas and transaction records displayed. "
        "Feature extraction (Slice 3) and baseline scoring model (Slice 5) "
        "will be implemented in subsequent slices."
    )


if __name__ == "__main__":
    main()
