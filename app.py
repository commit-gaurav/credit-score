"""AltScore AI — Unified FinTech Credit Intelligence Dashboard.

Integrates three core user experience designs into one seamless platform:
1. Design A (The Trust Passbook): Consumer-friendly, mobile-optimized, warm,
   and jargon-free financial health view with clean status badges, eligible
   rupee loan limits, and visual progress meters.
2. Design B (The Dual-Mode Switcher): Persistent top navigation bar allowing
   seamless toggling between the "Borrower View" and the "Bank Underwriter
   & Auditor View".
3. Design C (The Interactive What-If Coach): Real-time interactive sliders in
   the sidebar recalculating Repayment Probability, Illustrative Score (300-900),
   Risk Classification, and Eligible Loan Limit side-by-side with baseline.
"""

from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st

from data.personas import get_personas
from data.synthetic_data import (
    DEFAULT_ROW_COUNT,
    DEFAULT_SEED,
    generate_synthetic_dataset,
)
from fairness.audit import (
    DEFAULT_AUDIT_APPROVAL_THRESHOLD,
    calculate_fairness_metrics,
)
from features.extraction import extract_features
from model.scoring import (
    categorize_risk,
    explain_prediction,
    predict_repayment_probability,
    probability_to_score,
    train_model,
)

# -----------------------------------------------------------------------------
# Custom Fintech CSS Styling
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Global App Polish */
.stApp {
    background-color: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Header Banner */
.alt-header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 24px 28px;
    border-radius: 14px;
    color: #ffffff;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
}
.alt-header-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.02em;
}
.alt-header-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-top: 4px;
}

/* Premium Card Styles */
.alt-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    margin-bottom: 20px;
}
.alt-card-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 12px;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 8px;
}

/* Status Badges */
.badge-low-risk {
    background-color: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}
.badge-med-risk {
    background-color: #fffbeb;
    color: #92400e;
    border: 1px solid #fde68a;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}
.badge-high-risk {
    background-color: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}
.badge-neutral {
    background-color: #f1f5f9;
    color: #334155;
    border: 1px solid #cbd5e1;
    padding: 3px 10px;
    border-radius: 9999px;
    font-weight: 500;
    font-size: 0.80rem;
    display: inline-block;
}

/* Large Score Highlight */
.score-hero-number {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
}
.score-scale-text {
    font-size: 1.1rem;
    color: #64748b;
    font-weight: 500;
}

/* Custom Progress Meter */
.meter-bg {
    background-color: #e2e8f0;
    border-radius: 8px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    margin: 8px 0 12px 0;
}
.meter-fill-emerald {
    background-color: #10b981;
    height: 100%;
    border-radius: 8px;
}
.meter-fill-amber {
    background-color: #f59e0b;
    height: 100%;
    border-radius: 8px;
}
.meter-fill-coral {
    background-color: #ef4444;
    height: 100%;
    border-radius: 8px;
}

/* Plain English Explainer Card */
.explainer-box {
    background-color: #f8fafc;
    border-left: 4px solid #3b82f6;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin-top: 10px;
    font-size: 0.88rem;
    color: #334155;
}
</style>
"""


# -----------------------------------------------------------------------------
# Business Logic Helpers
# -----------------------------------------------------------------------------
def calculate_eligible_loan_limit(score: int, probability: float, liquid_balance: float) -> Dict[str, Any]:
    """Calculate an illustrative pre-approved micro-loan limit in Indian Rupees (₹).

    Grounds credit access in estimated repayment odds and liquid reserves:
    - Low Risk (Score >= 810): 3x - 5x monthly surplus (₹50,000 – ₹1,20,000)
    - Medium Risk (Score 720 - 809): 1.5x - 2.5x monthly surplus (₹20,000 – ₹50,000)
    - High Risk (Score < 720): Starter Micro-Credit line (₹10,000 Credit Builder)

    Args:
        score: Illustrative score in [300, 900].
        probability: Model estimated repayment probability.
        liquid_balance: Persona current liquid reserve.

    Returns:
        Dict[str, Any]: Formatted limit, raw amount, interest rate, and status badge.
    """
    if score >= 810:
        base_limit = 75000.0
        calculated = base_limit + ((score - 810) * 450.0)
        limit_val = int(min(120000, max(50000, round(calculated, -3))))
        interest_rate = "10.5% p.a."
        tenure = "12 – 24 Months"
        badge_text = "Pre-Approved"
        badge_class = "badge-low-risk"
    elif score >= 720:
        base_limit = 30000.0
        calculated = base_limit + ((score - 720) * 220.0)
        limit_val = int(min(50000, max(20000, round(calculated, -3))))
        interest_rate = "13.5% p.a."
        tenure = "6 – 12 Months"
        badge_text = "Eligible with Review"
        badge_class = "badge-med-risk"
    else:
        limit_val = 10000
        interest_rate = "15.0% p.a."
        tenure = "3 – 6 Months"
        badge_text = "Credit Builder Line"
        badge_class = "badge-high-risk"

    return {
        "limit": limit_val,
        "formatted_limit": f"₹{limit_val:,}",
        "interest_rate": interest_rate,
        "tenure": tenure,
        "badge_text": badge_text,
        "badge_class": badge_class,
    }


def render_dataframe(df: pd.DataFrame, hide_index: bool = False) -> None:
    """Render a DataFrame using modern width styling with backward compatibility."""
    try:
        st.dataframe(df, width="stretch", hide_index=hide_index)
    except (TypeError, ValueError):
        st.dataframe(df, use_container_width=True, hide_index=hide_index)


def get_risk_badge_html(category: str) -> str:
    """Return colored HTML status badge based on risk category."""
    if category == "Low Risk":
        return '<span class="badge-low-risk">● Low Risk (Approved)</span>'
    elif category == "Medium Risk":
        return '<span class="badge-med-risk">▲ Medium Risk (Manageable)</span>'
    else:
        return '<span class="badge-high-risk">■ High Risk (Needs Review)</span>'


# -----------------------------------------------------------------------------
# UI Component: Header & Prototype Disclaimer
# -----------------------------------------------------------------------------
def render_header() -> None:
    """Render the application top banner with role and prototype context."""
    st.markdown(
        """
        <div class="alt-header-banner">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1 class="alt-header-title">AltScore AI</h1>
                    <div class="alt-header-subtitle">
                        Explainable Alternative Credit Intelligence & Algorithmic Fairness Platform
                    </div>
                </div>
                <div style="margin-top: 8px;">
                    <span class="badge-neutral" style="background: rgba(255,255,255,0.15); color: #ffffff; border: 1px solid rgba(255,255,255,0.25);">
                        Synthetic Research Prototype • Non-Lending Tool
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer_banner() -> None:
    """Display the mandatory educational disclaimer."""
    st.caption(
        "⚠️ **PROTOTYPE NOTICE:** AltScore AI is an educational and research prototype. All borrower "
        "profiles, cash-flow ledgers, scores, and fairness metrics are demonstrative simulations. "
        "It does **not** access real consumer banking records and must not be used for actual lending decisions."
    )


# -----------------------------------------------------------------------------
# Design A: The Consumer Trust Passbook (Borrower Lens)
# -----------------------------------------------------------------------------
def render_borrower_passbook_view(
    persona: Dict[str, Any],
    features: Dict[str, float],
    probability: float,
    score: int,
    risk_category: str,
    loan_offer: Dict[str, Any],
    is_simulated: bool = False,
    baseline_score: int = 0,
    baseline_limit: int = 0,
) -> None:
    """Render the warm, jargon-free Trust Passbook view for consumer borrowers."""
    # Profile & Identity Overview
    st.markdown(
        f"""
        <div class="alt-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.82rem; text-transform: uppercase; color: #64748b; font-weight: 700; letter-spacing: 0.05em;">
                        Applicant Digital Passbook
                    </span>
                    <h2 style="margin: 2px 0 6px 0; color: #0f172a; font-size: 1.5rem;">{persona['name']}</h2>
                    <span style="color: #475569; font-size: 0.95rem;">{persona['role']}</span>
                </div>
                <div style="margin-top: 8px; text-align: right;">
                    <span class="badge-neutral" style="margin-right: 6px;">Geography: {persona['demographics']['geography']}</span>
                    <span class="badge-neutral">Liquid Savings: ₹{persona['liquid_balance']:,.2f}</span>
                </div>
            </div>
            <p style="margin-top: 12px; color: #334155; font-size: 0.92rem; line-height: 1.5;">
                {persona['summary']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Credit Health Card & Eligible Micro-Loan Limit
    col_score, col_limit = st.columns([1.1, 1.3])

    with col_score:
        score_color = "#059669" if score >= 810 else ("#d97706" if score >= 720 else "#dc2626")
        progress_pct = max(0, min(100, int(((score - 300) / 600) * 100)))
        fill_class = "meter-fill-emerald" if score >= 810 else ("meter-fill-amber" if score >= 720 else "meter-fill-coral")
        badge_html = get_risk_badge_html(risk_category)

        st.markdown(
            f"""
            <div class="alt-card" style="height: 100%;">
                <div class="alt-card-header">Credit Health Score</div>
                <div style="display: flex; align-items: baseline; gap: 8px;">
                    <span class="score-hero-number" style="color: {score_color};">{score}</span>
                    <span class="score-scale-text">/ 900</span>
                </div>
                <div style="margin: 8px 0;">{badge_html}</div>
                <div class="meter-bg">
                    <div class="{fill_class}" style="width: {progress_pct}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #94a3b8;">
                    <span>300 (Needs Work)</span>
                    <span>720 (Fair)</span>
                    <span>900 (Excellent)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_limit:
        st.markdown(
            f"""
            <div class="alt-card" style="height: 100%;">
                <div class="alt-card-header">Pre-Approved Micro-Credit Line</div>
                <div style="display: flex; align-items: baseline; gap: 10px;">
                    <span style="font-size: 2.3rem; font-weight: 800; color: #0f172a;">{loan_offer['formatted_limit']}</span>
                    <span class="{loan_offer['badge_class']}">{loan_offer['badge_text']}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px;">
                    <div style="background: #f8fafc; padding: 10px 14px; border-radius: 8px; border: 1px solid #e2e8f0;">
                        <span style="font-size: 0.78rem; color: #64748b; font-weight: 600;">ESTIMATED INTEREST</span>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">{loan_offer['interest_rate']}</div>
                    </div>
                    <div style="background: #f8fafc; padding: 10px 14px; border-radius: 8px; border: 1px solid #e2e8f0;">
                        <span style="font-size: 0.78rem; color: #64748b; font-weight: 600;">REPAYMENT TENURE</span>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">{loan_offer['tenure']}</div>
                    </div>
                </div>
                <p style="font-size: 0.82rem; color: #64748b; margin-top: 10px; margin-bottom: 0;">
                    Instant UPI disbursement directly to registered bank account upon confirmation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Daily Financial Health Progress Meters (4 Behavioral Habits)
    st.markdown("### Daily Financial Health Habits")
    st.caption("How your daily transaction behavior contributes to your credit standing without formal credit cards:")

    m1, m2 = st.columns(2)
    with m1:
        # 1. Income Flow Consistency
        inc_pct = int(features["income_stability"] * 100)
        inc_fill = "meter-fill-emerald" if inc_pct >= 80 else ("meter-fill-amber" if inc_pct >= 60 else "meter-fill-coral")
        st.markdown(
            f"""
            <div class="alt-card">
                <div style="display: flex; justify-content: space-between; font-weight: 600;">
                    <span style="color: #0f172a;">Income Flow Reliability</span>
                    <span style="color: #059669;">{inc_pct}% Score</span>
                </div>
                <div class="meter-bg">
                    <div class="{inc_fill}" style="width: {inc_pct}%;"></div>
                </div>
                <div style="font-size: 0.84rem; color: #475569;">
                    Evaluates consistency of month-over-month cash inflows. Stable gig or shop income builds trust.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Bill-Payment Discipline
        bill_pct = int(features["bill_payment_consistency"] * 100)
        bill_fill = "meter-fill-emerald" if bill_pct >= 90 else ("meter-fill-amber" if bill_pct >= 70 else "meter-fill-coral")
        st.markdown(
            f"""
            <div class="alt-card">
                <div style="display: flex; justify-content: space-between; font-weight: 600;">
                    <span style="color: #0f172a;">On-Time Bill Discipline</span>
                    <span style="color: #059669;">{bill_pct}% On-Time</span>
                </div>
                <div class="meter-bg">
                    <div class="{bill_fill}" style="width: {bill_pct}%;"></div>
                </div>
                <div style="font-size: 0.84rem; color: #475569;">
                    Tracks timely settlement of utility, mobile recharge, and rent payments.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        # 3. Monthly Activity Regularity
        reg_pct = int(features["payment_regularity"] * 100)
        reg_fill = "meter-fill-emerald" if reg_pct >= 80 else ("meter-fill-amber" if reg_pct >= 60 else "meter-fill-coral")
        st.markdown(
            f"""
            <div class="alt-card">
                <div style="display: flex; justify-content: space-between; font-weight: 600;">
                    <span style="color: #0f172a;">Activity Regularity</span>
                    <span style="color: #059669;">{reg_pct}% Active</span>
                </div>
                <div class="meter-bg">
                    <div class="{reg_fill}" style="width: {reg_pct}%;"></div>
                </div>
                <div style="font-size: 0.84rem; color: #475569;">
                    Checks for regular transaction inflows each month, proving active trade or gig employment.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4. Emergency Cushion
        buffer_val = features["financial_liquidity_buffer"]
        buf_pct = min(100, int((buffer_val / 6.0) * 100))
        buf_fill = "meter-fill-emerald" if buffer_val >= 3.0 else ("meter-fill-amber" if buffer_val >= 1.5 else "meter-fill-coral")
        st.markdown(
            f"""
            <div class="alt-card">
                <div style="display: flex; justify-content: space-between; font-weight: 600;">
                    <span style="color: #0f172a;">Emergency Savings Cushion</span>
                    <span style="color: #059669;">{buffer_val:.1f} Months</span>
                </div>
                <div class="meter-bg">
                    <div class="{buf_fill}" style="width: {buf_pct}%;"></div>
                </div>
                <div style="font-size: 0.84rem; color: #475569;">
                    Number of months your current savings can sustain living expenses (capped at 6 months).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Actionable Coaching Tips Card
    with st.expander("💡 Actionable Tips to Increase Your Credit Limit"):
        st.markdown(
            """
            - **Keep a 2-Month Buffer:** Storing even ₹5,000 more in your savings account significantly boosts your score.
            - **Never Delay Utility Bills:** Paying utility and phone bills on or before the due date keeps your bill discipline score at 100%.
            - **Regular Digital Inflows:** Depositing cash collections into your UPI account weekly proves steady income.
            """
        )

    # Passbook Transaction Records
    with st.expander("📋 View Recent 6-Month Passbook Activity"):
        df = pd.DataFrame(persona["transactions"]).rename(
            columns={
                "date": "Date",
                "description": "Description",
                "category": "Category",
                "type": "Type",
                "amount": "Amount (₹)",
                "status": "Status",
            }
        )
        df["Amount (₹)"] = df["Amount (₹)"].apply(lambda amt: f"₹{amt:,.2f}")
        render_dataframe(df, hide_index=True)


# -----------------------------------------------------------------------------
# Design B: Bank Underwriter & Risk Auditor View
# -----------------------------------------------------------------------------
def render_bank_underwriter_view(
    persona: Dict[str, Any],
    features: Dict[str, float],
    model: LogisticRegression,
    eval_metrics: Dict[str, Any],
    df_synthetic: pd.DataFrame,
    probability: float,
    score: int,
    risk_category: str,
) -> None:
    """Render the professional risk management, local explainability, and fairness audit panel."""
    st.markdown(
        """
        <div style="background: #0f172a; color: #f8fafc; padding: 16px 20px; border-radius: 10px; margin-bottom: 20px;">
            <div style="font-size: 0.80rem; text-transform: uppercase; color: #38bdf8; font-weight: 700; letter-spacing: 0.05em;">
                Institutional Risk Underwriting Console
            </div>
            <div style="font-size: 1.15rem; font-weight: 600; margin-top: 2px;">
                Algorithmic Risk Assessment & Demographic Compliance Audit
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Underwriting Decision Card
    u_col1, u_col2, u_col3 = st.columns(3)
    with u_col1:
        st.metric(
            label="Calibrated Repayment Probability",
            value=f"{probability:.2%}",
            help="Logistic sigmoid output P(repaid=1) based strictly on behavioral features.",
        )
    with u_col2:
        st.metric(
            label="Illustrative Bureau Score",
            value=str(score),
            help="Linear transformation: 300 + 600 × P(repayment), clamped to [300, 900].",
        )
    with u_col3:
        st.metric(
            label="Underwriter Risk Tier",
            value=risk_category,
            help="Tier based on documented probability thresholds (Low >=85%, Med 70-85%, High <70%).",
        )

    st.divider()

    # 2. Local Linear Log-Odds Decomposition (Waterfall representation)
    st.markdown("### 1. Local Prediction Explanation (Linear Decomposition)")
    st.caption(
        "Decomposes the applicant's prediction into exact feature-level log-odds contributions: "
        r"$\text{Logit} = \beta_0 + \sum \beta_i x_i$."
    )

    explanation = explain_prediction(model, features)

    # Table of local contributions
    contrib_rows = []
    for item in explanation["contributions"]:
        contrib_rows.append(
            {
                "Feature Name": item["display_name"],
                "Applicant Value (x)": f"{item['value']:.4f}",
                "Model Weight (β)": f"{item['weight']:+.4f}",
                "Local Contribution (β × x)": f"{item['contribution']:+.4f}",
                "Direction": "Positive (+)" if item["contribution"] >= 0 else "Negative (-)",
            }
        )
    render_dataframe(pd.DataFrame(contrib_rows), hide_index=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Base Model Intercept (β₀)", f"{explanation['intercept']:+.4f}")
    with c2:
        st.metric("Net Total Logit (β₀ + Σ βᵢxᵢ)", f"{explanation['total_logit']:+.4f}")
    with c3:
        st.metric("Sigmoid Output P", f"{explanation['implied_probability']:.2%}")

    # Plain English explanation card for accessibility
    st.markdown(
        """
        <div class="explainer-box">
            <strong>Plain English Explanation for Auditors:</strong><br/>
            - <strong>Model Weight (β):</strong> How sensitive the model is to each factor in general.<br/>
            - <strong>Local Contribution (β × x):</strong> The exact push this applicant receives. A positive number pushes the decision toward approval; a negative number reduces repayment odds.<br/>
            - <strong>Zero Demographic Leakage:</strong> Gender and geography columns were completely excluded from the model feature matrix.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # 3. Model Architecture & Held-Out Test Evaluation
    st.markdown("### 2. Baseline Model Performance (25% Held-Out Test Split)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("ROC-AUC", f"{eval_metrics['roc_auc']:.4f}")
    with m2:
        st.metric("Accuracy", f"{eval_metrics['accuracy']:.1%}")
    with m3:
        st.metric("Precision", f"{eval_metrics['precision']:.1%}")
    with m4:
        st.metric("Recall", f"{eval_metrics['recall']:.1%}")

    with st.expander("🔍 Inspect Global Logistic Regression Weights"):
        weights_df = pd.DataFrame(
            [
                {"Feature": col, "Learned Weight (β)": coef}
                for col, coef in eval_metrics["coefficients"].items()
            ]
        )
        render_dataframe(weights_df, hide_index=True)

    st.divider()

    # 4. Algorithmic Fairness & Disparate Impact Audit
    st.markdown("### 3. Automated Algorithmic Fairness & Disparate Impact Audit")
    st.write(
        f"Evaluates model approval rates across **Gender** and **Urban/Rural Geography** on the synthetic "
        f"evaluation dataset (N = {len(df_synthetic):,}) using documented approval threshold **P ≥ 70.0%** (Score ≥ 720)."
    )

    audit_results = calculate_fairness_metrics(
        model=model,
        evaluation_data=df_synthetic,
        threshold=DEFAULT_AUDIT_APPROVAL_THRESHOLD,
    )

    col_g, col_geo = st.columns(2)
    with col_g:
        g_info = audit_results["dimensions"]["gender"]
        st.markdown(
            f"""
            <div class="alt-card">
                <div class="alt-card-header">Dimension 1: Gender Audit</div>
                <div>• Reference Group (Higher Rate): <strong>{g_info['reference_group']} ({g_info['reference_rate']:.1%})</strong></div>
                <div>• Comparison Group (Lower Rate): <strong>{g_info['comparison_group']} ({g_info['comparison_rate']:.1%})</strong></div>
                <div style="margin-top: 14px;">
                    <span style="font-size: 0.82rem; color: #64748b; font-weight: 600;">DISPARATE IMPACT RATIO:</span>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #0f172a;">{g_info['ratio_display']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_geo:
        geo_info = audit_results["dimensions"]["geography"]
        st.markdown(
            f"""
            <div class="alt-card">
                <div class="alt-card-header">Dimension 2: Geography Audit</div>
                <div>• Reference Group (Higher Rate): <strong>{geo_info['reference_group']} ({geo_info['reference_rate']:.1%})</strong></div>
                <div>• Comparison Group (Lower Rate): <strong>{geo_info['comparison_group']} ({geo_info['comparison_rate']:.1%})</strong></div>
                <div style="margin-top: 14px;">
                    <span style="font-size: 0.82rem; color: #64748b; font-weight: 600;">DISPARATE IMPACT RATIO:</span>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #0f172a;">{geo_info['ratio_display']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Full audit summary table
    audit_table_rows = []
    for dim_key, dim_info in audit_results["dimensions"].items():
        audit_table_rows.append(
            {
                "Dimension": dim_info["dimension_name"],
                "Reference Group": f"{dim_info['reference_group']} ({dim_info['reference_rate']:.1%})",
                "Comparison Group": f"{dim_info['comparison_group']} ({dim_info['comparison_rate']:.1%})",
                "Disparate Impact Ratio": dim_info["ratio_display"],
                "Four-Fifths Rule (≥ 0.80)": (
                    "Pass (Synthetic)"
                    if dim_info["disparate_impact_ratio"] is not None and dim_info["disparate_impact_ratio"] >= 0.80
                    else "Disparity Flag"
                ),
                "Zero-Rate Check": "Valid (Reference Rate > 0%)" if dim_info["reference_rate"] > 0 else "Undefined",
            }
        )
    render_dataframe(pd.DataFrame(audit_table_rows), hide_index=True)

    # Proxy bias caveat
    st.warning(
        "⚠️ **CRITICAL GOVERNANCE NOTICE — THE PROXY BIAS DILEMMA:**  \n\n"
        "Excluding demographic attributes (`gender`, `geography`) from model inputs successfully eliminates "
        "**direct disparate treatment**. However, it does **not** protect against indirect proxy discrimination. "
        "Neutral financial signals (such as liquid savings or formal bill receipts) often reflect historical "
        "economic disparities. Algorithmic fairness requires continuous demographic outcome monitoring."
    )


# -----------------------------------------------------------------------------
# Design C: The Interactive What-If Coach (Sidebar & Comparison Panel)
# -----------------------------------------------------------------------------
def render_what_if_coach_sidebar(
    baseline_features: Dict[str, float],
    persona_name: str,
) -> Dict[str, float]:
    """Render real-time interactive sliders in the sidebar for scenario simulation."""
    st.sidebar.markdown("### 🎛️ Interactive What-If Coach")
    st.sidebar.caption(
        f"Simulate behavioral adjustments for **{persona_name}** to observe real-time "
        "recalculations in score, loan limit, and risk tier:"
    )

    adj_income = st.sidebar.slider(
        label="Income Flow Stability (0.00 – 1.00)",
        min_value=0.0,
        max_value=1.0,
        value=float(baseline_features["income_stability"]),
        step=0.01,
        key=f"coach_income_{persona_name}",
        help="Higher value indicates more consistent monthly earnings.",
    )

    adj_bills = st.sidebar.slider(
        label="Bill Payment Consistency (0.00 – 1.00)",
        min_value=0.0,
        max_value=1.0,
        value=float(baseline_features["bill_payment_consistency"]),
        step=0.01,
        key=f"coach_bills_{persona_name}",
        help="Proportion of scheduled bills paid on time.",
    )

    adj_regularity = st.sidebar.slider(
        label="Payment Regularity (0.00 – 1.00)",
        min_value=0.0,
        max_value=1.0,
        value=float(baseline_features["payment_regularity"]),
        step=0.01,
        key=f"coach_regularity_{persona_name}",
        help="Frequency of recorded monthly income credits.",
    )

    adj_liquidity = st.sidebar.slider(
        label="Emergency Liquidity Buffer (0.0 – 6.0 mos)",
        min_value=0.0,
        max_value=6.0,
        value=float(baseline_features["financial_liquidity_buffer"]),
        step=0.1,
        key=f"coach_liquidity_{persona_name}",
        help="Months of essential living expenses covered by savings.",
    )

    return {
        "income_stability": adj_income,
        "bill_payment_consistency": adj_bills,
        "payment_regularity": adj_regularity,
        "financial_liquidity_buffer": adj_liquidity,
    }


def render_what_if_comparison_card(
    baseline_prob: float,
    baseline_score: int,
    baseline_category: str,
    baseline_loan: Dict[str, Any],
    sim_prob: float,
    sim_score: int,
    sim_category: str,
    sim_loan: Dict[str, Any],
    persona_name: str,
) -> None:
    """Render side-by-side comparison card between baseline and what-if simulation."""
    st.markdown("### 🔄 What-If Simulation Comparison")
    st.caption("Side-by-side comparison of baseline standing vs. simulated behaviors:")

    c_base, c_sim = st.columns(2)

    with c_base:
        b_badge = get_risk_badge_html(baseline_category)
        st.markdown(
            f"""
            <div class="alt-card">
                <div class="alt-card-header">Baseline Standing ({persona_name})</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #1e293b;">{baseline_score} <span style="font-size: 1rem; color: #64748b;">/ 900</span></div>
                <div style="margin: 6px 0;">{b_badge}</div>
                <div style="margin-top: 10px; font-size: 0.90rem;">
                    <div>• Repayment Probability: <strong>{baseline_prob:.1%}</strong></div>
                    <div>• Eligible Credit Line: <strong>{baseline_loan['formatted_limit']}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_sim:
        score_diff = sim_score - baseline_score
        limit_diff = sim_loan["limit"] - baseline_loan["limit"]
        diff_color = "#059669" if score_diff > 0 else ("#dc2626" if score_diff < 0 else "#64748b")
        diff_text = f"+{score_diff} pts" if score_diff > 0 else (f"{score_diff} pts" if score_diff < 0 else "No change")
        limit_diff_text = f"+₹{limit_diff:,}" if limit_diff > 0 else (f"-₹{abs(limit_diff):,}" if limit_diff < 0 else "No change")
        s_badge = get_risk_badge_html(sim_category)

        st.markdown(
            f"""
            <div class="alt-card" style="border: 2px solid #3b82f6;">
                <div class="alt-card-header" style="color: #1d4ed8;">Simulated Outcome (Adjusted Inputs)</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #1e293b;">
                    {sim_score} <span style="font-size: 1rem; color: {diff_color}; font-weight: 700;">({diff_text})</span>
                </div>
                <div style="margin: 6px 0;">{s_badge}</div>
                <div style="margin-top: 10px; font-size: 0.90rem;">
                    <div>• Repayment Probability: <strong>{sim_prob:.1%}</strong> ({sim_prob - baseline_prob:+.1%})</div>
                    <div>• Eligible Credit Line: <strong>{sim_loan['formatted_limit']}</strong> ({limit_diff_text})</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Main Application Dashboard
# -----------------------------------------------------------------------------
def main() -> None:
    """Build the responsive AltScore AI dashboard with all three unified design views."""
    st.set_page_config(
        page_title="AltScore AI — Alternative Credit & Fairness Platform",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject Custom Fintech CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # 1. Render Top Header Banner
    render_header()
    render_disclaimer_banner()

    # 2. Load In-Memory Data and Model
    personas = get_personas()
    df_synthetic = generate_synthetic_dataset(seed=DEFAULT_SEED, row_count=DEFAULT_ROW_COUNT)
    model, eval_metrics = train_model(df_synthetic, seed=DEFAULT_SEED)

    # 3. Sidebar: Applicant Selector & What-If Coach Controls
    st.sidebar.markdown("### 👤 Select Applicant Profile")
    persona_keys = list(personas.keys())
    selected_key = st.sidebar.selectbox(
        label="Choose a thin-file borrower archetype:",
        options=persona_keys,
        format_func=lambda k: f"{personas[k]['name']} — {personas[k]['role']}",
    )
    selected_persona = personas[selected_key]

    # Extract Baseline Features
    baseline_features = extract_features(
        transactions=selected_persona["transactions"],
        liquid_balance=selected_persona["liquid_balance"],
    )

    # What-If Coach Sliders (Design C)
    st.sidebar.divider()
    sim_features = render_what_if_coach_sidebar(baseline_features, selected_persona["name"])

    # Baseline Calculations
    base_prob = predict_repayment_probability(model, baseline_features)
    base_score = probability_to_score(base_prob)
    base_cat = categorize_risk(base_prob)
    base_loan = calculate_eligible_loan_limit(base_score, base_prob, selected_persona["liquid_balance"])

    # Simulated Calculations (Real-time reactive)
    sim_prob = predict_repayment_probability(model, sim_features)
    sim_score = probability_to_score(sim_prob)
    sim_cat = categorize_risk(sim_prob)
    sim_loan = calculate_eligible_loan_limit(sim_score, sim_prob, selected_persona["liquid_balance"])

    # Detect if user moved sliders
    is_slider_adjusted = sim_features != baseline_features

    # 4. Design B: Persistent Dual-Mode Switcher Navigation Bar
    st.markdown("---")
    view_mode = st.radio(
        label="Operational Lens Switcher",
        options=[
            "Consumer Trust Passbook (Borrower Lens)",
            "Institutional Underwriting & Fairness Audit (Bank Lens)",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # 5. Render Selected Mode View
    if view_mode == "Consumer Trust Passbook (Borrower Lens)":
        # Passbook View
        active_score = sim_score if is_slider_adjusted else base_score
        active_prob = sim_prob if is_slider_adjusted else base_prob
        active_cat = sim_cat if is_slider_adjusted else base_cat
        active_features = sim_features if is_slider_adjusted else baseline_features
        active_loan = sim_loan if is_slider_adjusted else base_loan

        render_borrower_passbook_view(
            persona=selected_persona,
            features=active_features,
            probability=active_prob,
            score=active_score,
            risk_category=active_cat,
            loan_offer=active_loan,
            is_simulated=is_slider_adjusted,
            baseline_score=base_score,
            baseline_limit=base_loan["limit"],
        )

        # If sliders adjusted, display the comparison card
        if is_slider_adjusted:
            render_what_if_comparison_card(
                baseline_prob=base_prob,
                baseline_score=base_score,
                baseline_category=base_cat,
                baseline_loan=base_loan,
                sim_prob=sim_prob,
                sim_score=sim_score,
                sim_category=sim_cat,
                sim_loan=sim_loan,
                persona_name=selected_persona["name"],
            )

    else:
        # Bank Underwriter & Risk Auditor View
        active_score = sim_score if is_slider_adjusted else base_score
        active_prob = sim_prob if is_slider_adjusted else base_prob
        active_cat = sim_cat if is_slider_adjusted else base_cat
        active_features = sim_features if is_slider_adjusted else baseline_features

        render_bank_underwriter_view(
            persona=selected_persona,
            features=active_features,
            model=model,
            eval_metrics=eval_metrics,
            df_synthetic=df_synthetic,
            probability=active_prob,
            score=active_score,
            risk_category=active_cat,
        )

        if is_slider_adjusted:
            st.divider()
            render_what_if_comparison_card(
                baseline_prob=base_prob,
                baseline_score=base_score,
                baseline_category=base_cat,
                baseline_loan=base_loan,
                sim_prob=sim_prob,
                sim_score=sim_score,
                sim_category=sim_cat,
                sim_loan=sim_loan,
                persona_name=selected_persona["name"],
            )

    # Footer
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #94a3b8; font-size: 0.82rem; padding: 12px 0;">
            AltScore AI • Hackathon Alternative Credit Scoring & Fairness Audit Prototype • Open Research Demo
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
