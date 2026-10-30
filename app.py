"""AltScore AI — Minimal Streamlit Application Shell.

Slice 1: Initializes the core dashboard with project title, purpose,
and mandatory synthetic research prototype disclaimer.
"""

import streamlit as st


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


def main() -> None:
    """Build the dashboard and render the initial application shell."""
    # st.set_page_config must be called as the first Streamlit command
    st.set_page_config(
        page_title="AltScore AI",
        page_icon="💳",
        layout="wide",
    )

    render_header()
    render_disclaimer()

    st.divider()
    st.info(
        "**Slice 1 Active:** App shell initialized. "
        "Personas, feature extraction, scoring model, what-if simulator, "
        "and fairness audit will be loaded in subsequent slices."
    )


if __name__ == "__main__":
    main()

