"""AltScore AI — Explainable Alternative Credit Intelligence & Algorithmic Fairness Platform.

Flagship FinTech credit intelligence web application designed for thin-file borrower evaluation.
Featuring:
- Mobile-First Architecture: Main-canvas interactive controls & What-If Coach (no sidebar trap)
- Hero Banner & Collapsible Research Prototype Notice
- 3-Card High-Contrast KPI Strip (Score, Probability, Pre-Approved Micro-Line in ₹)
- Native Touch-Friendly Tabs (Consumer Trust Passbook vs. Institutional Underwriting & Fairness Audit)
- 100% Native Container Card Isolation (Zero leaked HTML / <pre><code> boxes)
"""

from pathlib import Path
import sys
from typing import Any, Dict, Tuple

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


# =============================================================================
# 1. Custom CSS Injection (Bulletproof Contrast & Mobile-First Styling)
# =============================================================================
def inject_custom_css() -> None:
    """Inject flagship FinTech styling, dark mode normalization, and mobile responsiveness."""
    css = (
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');"
        ":root {"
        "--bg-canvas: #F1F5F9;"
        "--card-bg: #FFFFFF;"
        "--text-main: #0F172A;"
        "--text-muted: #64748B;"
        "--border-color: #E2E8F0;"
        "--accent-blue: #2563EB;"
        "--accent-emerald: #059669;"
        "}"
        ".stApp {"
        "background-color: var(--bg-canvas) !important;"
        "color: var(--text-main) !important;"
        "font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;"
        "}"
        ".main .block-container {"
        "max-width: 1200px !important;"
        "padding: 1.25rem 1.5rem 3rem 1.5rem !important;"
        "color: var(--text-main) !important;"
        "}"
        ".main p, .main span, .main label, .main div,"
        ".main div[data-testid='stMarkdownContainer'] p,"
        ".main div[data-testid='stMarkdownContainer'] span,"
        ".main div[data-testid='stMarkdownContainer'] li {"
        "color: var(--text-main) !important;"
        "}"
        ".main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {"
        "color: var(--text-main) !important;"
        "font-weight: 700 !important;"
        "letter-spacing: -0.02em;"
        "}"
        "[data-testid='stVerticalBlockBorderWrapper'] {"
        "background-color: var(--card-bg) !important;"
        "border: 1px solid var(--border-color) !important;"
        "border-radius: 16px !important;"
        "box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04) !important;"
        "padding: 1.25rem !important;"
        "margin-bottom: 1rem !important;"
        "transition: transform 0.2s ease, box-shadow 0.2s ease;"
        "overflow: hidden !important;"
        "word-wrap: break-word !important;"
        "}"
        "[data-testid='stVerticalBlockBorderWrapper']:hover {"
        "box-shadow: 0 8px 20px rgba(15, 23, 42, 0.07) !important;"
        "}"
        "button[data-baseweb='tab'] {"
        "background-color: #E2E8F0 !important;"
        "border-radius: 12px !important;"
        "padding: 10px 22px !important;"
        "margin-right: 10px !important;"
        "font-weight: 600 !important;"
        "color: #475569 !important;"
        "border: 1px solid transparent !important;"
        "min-height: 44px !important;"
        "transition: all 0.2s ease !important;"
        "}"
        "button[data-baseweb='tab']:hover {"
        "background-color: #CBD5E1 !important;"
        "color: #0F172A !important;"
        "}"
        "button[data-baseweb='tab'][aria-selected='true'] {"
        "background-color: #2563EB !important;"
        "color: #FFFFFF !important;"
        "box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;"
        "}"
        "button[data-baseweb='tab'] p {"
        "color: inherit !important;"
        "font-weight: inherit !important;"
        "font-size: 0.92rem !important;"
        "}"
        ".alt-hero-banner {"
        "background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);"
        "padding: 22px 28px;"
        "border-radius: 16px;"
        "color: #FFFFFF !important;"
        "margin-bottom: 14px;"
        "box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15);"
        "border: 1px solid rgba(255, 255, 255, 0.10);"
        "}"
        ".alt-hero-content {"
        "display: flex;"
        "justify-content: space-between;"
        "align-items: center;"
        "flex-wrap: wrap;"
        "gap: 12px;"
        "}"
        ".alt-hero-title {"
        "font-size: 1.85rem;"
        "font-weight: 800;"
        "color: #FFFFFF !important;"
        "margin: 0;"
        "letter-spacing: -0.025em;"
        "}"
        ".alt-hero-subtitle {"
        "font-size: 0.92rem;"
        "color: #94A3B8 !important;"
        "margin-top: 4px;"
        "}"
        ".alt-pill-badge {"
        "background: rgba(255, 255, 255, 0.12);"
        "color: #F8FAFC !important;"
        "border: 1px solid rgba(255, 255, 255, 0.22);"
        "padding: 6px 14px;"
        "border-radius: 9999px;"
        "font-size: 0.80rem;"
        "font-weight: 600;"
        "display: inline-block;"
        "letter-spacing: 0.02em;"
        "}"
        ".badge-container {"
        "display: flex;"
        "gap: 8px;"
        "flex-wrap: wrap;"
        "margin-top: 10px;"
        "}"
        ".badge-neutral {"
        "background-color: #F1F5F9 !important;"
        "color: #334155 !important;"
        "border: 1px solid #CBD5E1 !important;"
        "padding: 4px 12px;"
        "border-radius: 9999px;"
        "font-weight: 600;"
        "font-size: 0.80rem;"
        "display: inline-block;"
        "}"
        ".badge-low-risk {"
        "background-color: #ECFDF5 !important;"
        "color: #065F46 !important;"
        "border: 1px solid #A7F3D0 !important;"
        "padding: 4px 12px;"
        "border-radius: 9999px;"
        "font-weight: 700;"
        "font-size: 0.82rem;"
        "display: inline-block;"
        "}"
        ".badge-med-risk {"
        "background-color: #FFFBEB !important;"
        "color: #92400E !important;"
        "border: 1px solid #FDE68A !important;"
        "padding: 4px 12px;"
        "border-radius: 9999px;"
        "font-weight: 700;"
        "font-size: 0.82rem;"
        "display: inline-block;"
        "}"
        ".badge-high-risk {"
        "background-color: #FEF2F2 !important;"
        "color: #991B1B !important;"
        "border: 1px solid #FECACA !important;"
        "padding: 4px 12px;"
        "border-radius: 9999px;"
        "font-weight: 700;"
        "font-size: 0.82rem;"
        "display: inline-block;"
        "}"
        ".explainer-box {"
        "background-color: #F8FAFC !important;"
        "border-left: 4px solid var(--accent-blue) !important;"
        "padding: 14px 18px !important;"
        "border-radius: 0 10px 10px 0 !important;"
        "margin-top: 12px !important;"
        "font-size: 0.88rem !important;"
        "color: #334155 !important;"
        "border-top: 1px solid #E2E8F0;"
        "border-right: 1px solid #E2E8F0;"
        "border-bottom: 1px solid #E2E8F0;"
        "}"
        "header[data-testid='stHeader'] { background: transparent !important; }"
        ".stDeployButton, [data-testid='stToolbarActions'], footer, #MainMenu { display: none !important; }"
        "@media (max-width: 768px) {"
        ".main .block-container { padding: 0.75rem 0.75rem 3rem 0.75rem !important; }"
        ".alt-hero-banner { padding: 16px 18px !important; border-radius: 14px !important; }"
        ".alt-hero-title { font-size: 1.50rem !important; }"
        ".alt-hero-subtitle { font-size: 0.82rem !important; }"
        "button[data-baseweb='tab'] { width: 100% !important; margin-bottom: 8px !important; text-align: center !important; }"
        "}"
        "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# Business Logic & Loan Calculations
# =============================================================================
def calculate_eligible_loan_limit(score: int, probability: float, liquid_balance: float) -> Dict[str, Any]:
    """Calculate an illustrative pre-approved micro-loan limit in Indian Rupees (₹)."""
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
        return '<span class="badge-low-risk">● Low Risk • Prime Equivalent</span>'
    elif category == "Medium Risk":
        return '<span class="badge-med-risk">▲ Medium Risk • Near Prime</span>'
    else:
        return '<span class="badge-high-risk">■ High Risk • Micro Builder</span>'


# =============================================================================
# 2. Modular Component: Hero Banner & Collapsible Notice
# =============================================================================
def render_hero_banner() -> None:
    """Render the sleek Deep Navy Hero Banner."""
    st.markdown(
        '<div class="alt-hero-banner">'
        '<div class="alt-hero-content">'
        '<div>'
        '<h1 class="alt-hero-title">AltScore AI</h1>'
        '<div class="alt-hero-subtitle">Explainable Alternative Credit Intelligence & Algorithmic Fairness Platform</div>'
        '</div>'
        '<div><span class="alt-pill-badge">Synthetic Research Prototype • Non-Lending Tool</span></div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    """Render a sleek, single-line collapsible notice that does not clutter the screen."""
    with st.expander("ℹ️ Prototype & Non-Lending Notice (Click to inspect governance details)"):
        st.markdown(
            "**Mandatory Research Prototype Notice:** AltScore AI is an educational and research prototype. "
            "All borrower profiles, financial transactions, scores, and fairness metrics are entirely synthetic simulations. "
            "This tool does **not** access real consumer banking records, does not produce legal credit bureau scores "
            "(e.g., CIBIL/FICO), and must not be used for actual lending decisions."
        )


# =============================================================================
# 3. Main-Canvas Interactive Control Bar & What-If Coach (Mobile-Friendly)
# =============================================================================
def render_control_bar_and_coach(
    personas: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float], bool]:
    """Render the applicant profile selector and What-If coach right on the main canvas.

    Allows immediate mobile interaction without trapping users in a sidebar.
    """
    with st.container(border=True):
        col_profile, col_coach = st.columns([1.1, 2.0], gap="large")

        with col_profile:
            st.markdown("#### 👤 Select Applicant Profile")
            persona_keys = list(personas.keys())
            selected_key = st.selectbox(
                label="Choose a thin-file borrower archetype:",
                options=persona_keys,
                format_func=lambda k: f"{personas[k]['name']} — {personas[k]['role'].split('(')[0].strip()}",
                label_visibility="collapsed",
            )
            selected_persona = personas[selected_key]

            # Neat flex-wrap badges under selector
            st.markdown(
                '<div class="badge-container">'
                f'<span class="badge-neutral">Role: {selected_persona["role"].split("(")[0].strip()}</span>'
                f'<span class="badge-neutral">Region: {selected_persona["demographics"]["geography"]} ({selected_persona["demographics"]["gender"]})</span>'
                f'<span class="badge-neutral">Savings: ₹{selected_persona["liquid_balance"]:,.2f}</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.caption(selected_persona["summary"])

        # Extract baseline features
        baseline_features = extract_features(
            transactions=selected_persona["transactions"],
            liquid_balance=selected_persona["liquid_balance"],
        )

        with col_coach:
            st.markdown("#### 🎛️ Interactive What-If Scenario Coach")
            st.caption("Adjust behavioral cash-flow habits to simulate score and credit limit shifts in real time:")

            c1, c2 = st.columns(2)
            with c1:
                adj_income = st.slider(
                    label="Income Flow Stability",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(baseline_features["income_stability"]),
                    step=0.01,
                    key=f"main_inc_{selected_key}",
                    help="Monthly income consistency (1.0 = steady).",
                )
                adj_bills = st.slider(
                    label="Bill Payment Consistency",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(baseline_features["bill_payment_consistency"]),
                    step=0.01,
                    key=f"main_bill_{selected_key}",
                    help="Fraction of scheduled bills paid on time.",
                )
            with c2:
                adj_reg = st.slider(
                    label="Payment Regularity",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(baseline_features["payment_regularity"]),
                    step=0.01,
                    key=f"main_reg_{selected_key}",
                    help="Frequency of monthly incoming transactions.",
                )
                adj_liq = st.slider(
                    label="Emergency Buffer (Months)",
                    min_value=0.0,
                    max_value=6.0,
                    value=float(baseline_features["financial_liquidity_buffer"]),
                    step=0.1,
                    key=f"main_liq_{selected_key}",
                    help="Months of living expenses covered by savings (capped at 6.0).",
                )

        sim_features = {
            "income_stability": adj_income,
            "bill_payment_consistency": adj_bills,
            "payment_regularity": adj_reg,
            "financial_liquidity_buffer": adj_liq,
        }

        is_slider_adjusted = sim_features != baseline_features

    return selected_persona, baseline_features, sim_features, is_slider_adjusted


# =============================================================================
# 4. Sticky / Prominent 3-Card KPI Strip
# =============================================================================
def render_sticky_kpis(
    probability: float,
    score: int,
    risk_category: str,
    loan_offer: Dict[str, Any],
    is_simulated: bool = False,
    baseline_score: int = 0,
    baseline_loan: Dict[str, Any] = None,
) -> None:
    """Render the 3 core real-time output cards in clean, equal-height containers."""
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        with st.container(border=True):
            score_color = "#059669" if score >= 810 else ("#D97706" if score >= 720 else "#DC2626")
            score_delta_html = ""
            if is_simulated and baseline_score != 0:
                diff = score - baseline_score
                sign = "+" if diff > 0 else ""
                d_color = "#059669" if diff > 0 else ("#DC2626" if diff < 0 else "#64748B")
                score_delta_html = f'<span style="font-size: 0.92rem; font-weight: 700; color: {d_color}; margin-left: 6px;">({sign}{diff} pts)</span>'

            st.markdown(
                '<div style="font-size: 0.78rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">'
                'ILLUSTRATIVE CREDIT SCORE'
                '</div>'
                '<div style="display: flex; align-items: baseline; gap: 4px; margin: 4px 0; white-space: nowrap;">'
                f'<span style="font-size: 2.3rem; font-weight: 800; color: {score_color}; line-height: 1;">{score}</span>'
                '<span style="font-size: 1.05rem; color: #64748B; font-weight: 600;">/ 900</span>'
                f'{score_delta_html}'
                '</div>'
                f'<div style="margin: 8px 0 10px 0;">{get_risk_badge_html(risk_category)}</div>',
                unsafe_allow_html=True,
            )
            pct = max(0.0, min(1.0, (score - 300) / 600))
            st.progress(pct)

    with kpi2:
        with st.container(border=True):
            st.markdown(
                '<div style="font-size: 0.78rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">'
                'REPAYMENT PROBABILITY'
                '</div>'
                f'<div style="font-size: 2.3rem; font-weight: 800; color: #0F172A; line-height: 1; margin: 4px 0;">'
                f'{probability:.1%}'
                '</div>'
                '<div style="margin: 8px 0;">'
                '<span class="badge-neutral">Calibrated Logistic Sigmoid</span>'
                '</div>'
                '<div style="font-size: 0.82rem; color: #64748B; margin-top: 6px;">'
                'Multi-factor cash-flow behavioral estimate'
                '</div>',
                unsafe_allow_html=True,
            )

    with kpi3:
        with st.container(border=True):
            limit_delta_html = ""
            if is_simulated and baseline_loan:
                l_diff = loan_offer["limit"] - baseline_loan["limit"]
                if l_diff != 0:
                    l_sign = "+" if l_diff > 0 else ""
                    l_col = "#059669" if l_diff > 0 else "#DC2626"
                    limit_delta_html = f'<span style="font-size: 0.90rem; font-weight: 700; color: {l_col}; margin-left: 6px;">({l_sign}₹{abs(l_diff):,})</span>'

            st.markdown(
                '<div style="font-size: 0.78rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">'
                'PRE-APPROVED MICRO-LINE'
                '</div>'
                f'<div style="display: flex; align-items: baseline; margin: 4px 0; white-space: nowrap;">'
                f'<span style="font-size: 2.3rem; font-weight: 800; color: #0F172A; line-height: 1;">{loan_offer["formatted_limit"]}</span>'
                f'{limit_delta_html}'
                '</div>'
                '<div style="margin: 8px 0;">'
                f'<span class="{loan_offer["badge_class"]}">{loan_offer["badge_text"]}</span>'
                '</div>'
                '<div style="font-size: 0.82rem; color: #64748B; margin-top: 6px;">'
                'Instant micro-line terms • UPI Disbursed'
                '</div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# 5. Tab 1: The Consumer Trust Passbook (Borrower Lens)
# =============================================================================
def render_trust_passbook_tab(
    persona: Dict[str, Any],
    features: Dict[str, float],
    score: int,
    risk_category: str,
    loan_offer: Dict[str, Any],
    is_simulated: bool = False,
    baseline_score: int = 0,
    baseline_limit: int = 0,
    baseline_features: Dict[str, float] = None,
) -> None:
    """Render Design A: The Warm, Consumer-Friendly Trust Passbook."""
    # Section 1: Credit Standing & Instant Micro-Line Terms
    with st.container(border=True):
        col_t1, col_t2 = st.columns([1.1, 1.3], gap="large")

        with col_t1:
            st.markdown("#### 💳 Credit Standing Summary")
            st.markdown(
                f"<div style='font-size: 1.05rem; color: #334155; margin-bottom: 8px;'>"
                f"Applicant <strong>{persona['name']}</strong> currently holds an illustrative score of "
                f"<strong>{score}</strong> out of 900."
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"Status: {get_risk_badge_html(risk_category)}", unsafe_allow_html=True)
            st.caption(
                "Your score is dynamically estimated from day-to-day cash-flow habits and liquid reserves, "
                "with zero dependency on traditional credit bureau cards."
            )

        with col_t2:
            st.markdown("#### ⚡ Pre-Approved Micro-Line Details")
            st.markdown(
                f"<div style='font-size: 2rem; font-weight: 800; color: #0F172A; margin-bottom: 8px;'>"
                f"{loan_offer['formatted_limit']}"
                f"</div>",
                unsafe_allow_html=True,
            )
            col_t2a, col_t2b = st.columns(2)
            with col_t2a:
                st.markdown(
                    "<div style='background: #F8FAFC; padding: 10px 14px; border-radius: 8px; border: 1px solid #E2E8F0;'>"
                    "<span style='font-size: 0.76rem; color: #64748B; font-weight: 700; text-transform: uppercase;'>ESTIMATED INTEREST</span>"
                    f"<div style='font-size: 1.1rem; font-weight: 700; color: #0F172A;'>{loan_offer['interest_rate']}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with col_t2b:
                st.markdown(
                    "<div style='background: #F8FAFC; padding: 10px 14px; border-radius: 8px; border: 1px solid #E2E8F0;'>"
                    "<span style='font-size: 0.76rem; color: #64748B; font-weight: 700; text-transform: uppercase;'>REPAYMENT TENURE</span>"
                    f"<div style='font-size: 1.1rem; font-weight: 700; color: #0F172A;'>{loan_offer['tenure']}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            st.caption("Disburses instantly via UPI directly to your registered bank account upon confirmation.")

    # Section 2: Daily Financial Health Habits
    with st.container(border=True):
        st.markdown("#### 📊 Daily Financial Health Habits")
        st.caption("How your daily transaction behavior contributes to your credit standing:")

        h1, h2 = st.columns(2, gap="large")

        with h1:
            # 1. Income Flow Stability
            inc_pct = int(features["income_stability"] * 100)
            st.markdown(f"**Income Flow Reliability:** `{inc_pct}% Score`")
            st.progress(features["income_stability"])
            st.caption("Consistency of month-over-month earnings. Stable gig or shop income builds institutional trust.")

            # 2. Bill-Payment Discipline
            bill_pct = int(features["bill_payment_consistency"] * 100)
            st.markdown(f"**On-Time Bill Discipline:** `{bill_pct}% On-Time`")
            st.progress(features["bill_payment_consistency"])
            st.caption("Timely settlement of electricity, mobile recharge, and rent payments.")

        with h2:
            # 3. Monthly Activity Regularity
            reg_pct = int(features["payment_regularity"] * 100)
            st.markdown(f"**Activity Regularity:** `{reg_pct}% Active Months`")
            st.progress(features["payment_regularity"])
            st.caption("Regular monthly transaction inflows proving active trade or gig employment.")

            # 4. Emergency Cushion
            buffer_val = features["financial_liquidity_buffer"]
            buf_pct = min(1.0, buffer_val / 6.0)
            st.markdown(f"**Emergency Savings Cushion:** `{buffer_val:.1f} Months`")
            st.progress(buf_pct)
            st.caption("Number of months current liquid savings can sustain living expenses (capped at 6.0 mos).")

    # Section 3: If What-If Sliders Adjusted, Show Side-by-Side Breakdown
    if is_simulated and baseline_features:
        with st.container(border=True):
            st.markdown("#### 🔄 What-If Simulation Comparison Breakdown")
            comp_rows = [
                {
                    "Metric": "Illustrative Credit Score",
                    "Baseline Value": f"{baseline_score} / 900",
                    "Simulated Value": f"{score} / 900",
                    "Net Shift": f"{score - baseline_score:+d} pts",
                },
                {
                    "Metric": "Pre-Approved Credit Line",
                    "Baseline Value": f"₹{baseline_limit:,}",
                    "Simulated Value": f"{loan_offer['formatted_limit']}",
                    "Net Shift": f"₹{loan_offer['limit'] - baseline_limit:+,}",
                },
                {
                    "Metric": "Income Flow Stability",
                    "Baseline Value": f"{baseline_features['income_stability']:.2f}",
                    "Simulated Value": f"{features['income_stability']:.2f}",
                    "Net Shift": f"{features['income_stability'] - baseline_features['income_stability']:+.2f}",
                },
                {
                    "Metric": "Bill Payment Consistency",
                    "Baseline Value": f"{baseline_features['bill_payment_consistency']:.2f}",
                    "Simulated Value": f"{features['bill_payment_consistency']:.2f}",
                    "Net Shift": f"{features['bill_payment_consistency'] - baseline_features['bill_payment_consistency']:+.2f}",
                },
                {
                    "Metric": "Emergency Liquidity Buffer",
                    "Baseline Value": f"{baseline_features['financial_liquidity_buffer']:.1f} mos",
                    "Simulated Value": f"{features['financial_liquidity_buffer']:.1f} mos",
                    "Net Shift": f"{features['financial_liquidity_buffer'] - baseline_features['financial_liquidity_buffer']:+.1f} mos",
                },
            ]
            render_dataframe(pd.DataFrame(comp_rows), hide_index=True)

    # Section 4: Actionable Tips
    with st.expander("💡 Actionable Tips to Unlock Higher Credit Limits"):
        st.markdown(
            "- **Maintain a 2-Month Savings Buffer:** Storing even ₹5,000 more in your savings account significantly boosts your score.\n"
            "- **Never Delay Utility Bills:** Paying utility and phone bills on or before the due date keeps your bill discipline score at 100%.\n"
            "- **Deposit Cash Digitally:** Depositing cash collections into your UPI account weekly proves steady income."
        )

    # Section 5: Recent 6-Month Passbook Activity
    with st.expander("📋 View Recent 6-Month Passbook Activity Ledger"):
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


# =============================================================================
# 6. Tab 2: Institutional Underwriting & Algorithmic Fairness Audit
# =============================================================================
def render_underwriter_tab(
    persona: Dict[str, Any],
    features: Dict[str, float],
    model: LogisticRegression,
    eval_metrics: Dict[str, Any],
    df_synthetic: pd.DataFrame,
    probability: float,
    score: int,
    risk_category: str,
) -> None:
    """Render Design B: The Professional Risk Management and Audit Console."""
    # Section 1: Underwriting Decision & Diagnostics
    with st.container(border=True):
        st.markdown("#### 🏦 Institutional Risk Underwriting Console")
        st.caption("Algorithmic Risk Assessment & Calibrated Logistic Scoring")

        u1, u2, u3 = st.columns(3)
        with u1:
            st.metric(
                label="Calibrated Repayment Probability",
                value=f"{probability:.2%}",
                help="Logistic sigmoid output P(repaid=1) based strictly on behavioral features.",
            )
        with u2:
            st.metric(
                label="Illustrative Bureau Score",
                value=str(score),
                help="Linear transformation: 300 + 600 × P(repayment), clamped to [300, 900].",
            )
        with u3:
            st.metric(
                label="Underwriter Risk Tier",
                value=risk_category,
                help="Tier based on documented probability thresholds (Low >=85%, Med 70-85%, High <70%).",
            )

    # Section 2: Local Linear Log-Odds Decomposition
    with st.container(border=True):
        st.markdown("#### 🔍 1. Local Prediction Explanation (Linear Log-Odds Decomposition)")
        st.caption(
            "Decomposes applicant prediction into exact feature-level log-odds contributions: "
            r"$\text{Logit} = \beta_0 + \sum \beta_i x_i$."
        )

        explanation = explain_prediction(model, features)
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

        st.markdown(
            '<div class="explainer-box">'
            '<strong>Plain English Explanation for Auditors:</strong><br/>'
            '• <strong>Model Weight (β):</strong> General sensitivity of the model across the training distribution.<br/>'
            '• <strong>Local Contribution (β × x):</strong> The exact log-odds shift this applicant receives. A positive number increases repayment odds; a negative number reduces them.<br/>'
            '• <strong>Zero Demographic Leakage:</strong> Gender and geography columns were completely excluded from the model feature matrix.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Section 3: Baseline Model Performance
    with st.container(border=True):
        st.markdown("#### 📈 2. Baseline Model Performance (25% Held-Out Test Split)")
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

    # Section 4: Algorithmic Fairness & Disparate Impact Audit
    with st.container(border=True):
        st.markdown("#### ⚖️ 3. Automated Algorithmic Fairness & Disparate Impact Audit")
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
            with st.container(border=True):
                st.markdown("##### Dimension 1: Gender Audit")
                st.write(f"• Reference Group (Higher Rate): **{g_info['reference_group']} ({g_info['reference_rate']:.1%})**")
                st.write(f"• Comparison Group (Lower Rate): **{g_info['comparison_group']} ({g_info['comparison_rate']:.1%})**")
                st.markdown(
                    "<span style='font-size: 0.82rem; color: #64748B; font-weight: 700;'>DISPARATE IMPACT RATIO:</span>"
                    f"<div style='font-size: 1.85rem; font-weight: 800; color: #0F172A;'>{g_info['ratio_display']}</div>",
                    unsafe_allow_html=True,
                )

        with col_geo:
            geo_info = audit_results["dimensions"]["geography"]
            with st.container(border=True):
                st.markdown("##### Dimension 2: Geography Audit")
                st.write(f"• Reference Group (Higher Rate): **{geo_info['reference_group']} ({geo_info['reference_rate']:.1%})**")
                st.write(f"• Comparison Group (Lower Rate): **{geo_info['comparison_group']} ({geo_info['comparison_rate']:.1%})**")
                st.markdown(
                    "<span style='font-size: 0.82rem; color: #64748B; font-weight: 700;'>DISPARATE IMPACT RATIO:</span>"
                    f"<div style='font-size: 1.85rem; font-weight: 800; color: #0F172A;'>{geo_info['ratio_display']}</div>",
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

        st.warning(
            "⚠️ **CRITICAL GOVERNANCE NOTICE — THE PROXY BIAS DILEMMA:**\n\n"
            "Excluding demographic attributes (`gender`, `geography`) from model inputs successfully eliminates "
            "**direct disparate treatment**. However, it does **not** protect against indirect proxy discrimination. "
            "Neutral financial signals (such as liquid savings or formal bill receipts) often reflect historical "
            "economic disparities. Algorithmic fairness requires continuous demographic outcome monitoring."
        )


# =============================================================================
# Main Application Entrypoint
# =============================================================================
def main() -> None:
    """Build and render the unified AltScore AI FinTech web application."""
    st.set_page_config(
        page_title="AltScore AI — Alternative Credit & Fairness Platform",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # 1. Inject Bulletproof CSS
    inject_custom_css()

    # 2. Render Hero Header & Collapsible Notice
    render_hero_banner()
    render_disclaimer()

    # 3. Load In-Memory Data & Train Baseline Model
    personas = get_personas()
    df_synthetic = generate_synthetic_dataset(seed=DEFAULT_SEED, row_count=DEFAULT_ROW_COUNT)
    model, eval_metrics = train_model(df_synthetic, seed=DEFAULT_SEED)

    # 4. Main-Canvas Interactive Control Bar & What-If Coach (Mobile-Friendly)
    selected_persona, baseline_features, sim_features, is_slider_adjusted = render_control_bar_and_coach(personas)

    # 5. Baseline Calculations
    base_prob = predict_repayment_probability(model, baseline_features)
    base_score = probability_to_score(base_prob)
    base_cat = categorize_risk(base_prob)
    base_loan = calculate_eligible_loan_limit(base_score, base_prob, selected_persona["liquid_balance"])

    # 6. Simulated Calculations (Reactive to What-If sliders)
    sim_prob = predict_repayment_probability(model, sim_features)
    sim_score = probability_to_score(sim_prob)
    sim_cat = categorize_risk(sim_prob)
    sim_loan = calculate_eligible_loan_limit(sim_score, sim_prob, selected_persona["liquid_balance"])

    active_score = sim_score if is_slider_adjusted else base_score
    active_prob = sim_prob if is_slider_adjusted else base_prob
    active_cat = sim_cat if is_slider_adjusted else base_cat
    active_features = sim_features if is_slider_adjusted else baseline_features
    active_loan = sim_loan if is_slider_adjusted else base_loan

    # 7. Sticky / Prominent 3-Card KPI Strip
    render_sticky_kpis(
        probability=active_prob,
        score=active_score,
        risk_category=active_cat,
        loan_offer=active_loan,
        is_simulated=is_slider_adjusted,
        baseline_score=base_score,
        baseline_loan=base_loan,
    )

    # 8. Native Touch-Friendly Navigation Tabs
    tab_borrower, tab_underwriter = st.tabs([
        "📱 Consumer Trust Passbook (Borrower Lens)",
        "🏦 Institutional Underwriting & Fairness Audit (Bank Lens)",
    ])

    with tab_borrower:
        render_trust_passbook_tab(
            persona=selected_persona,
            features=active_features,
            score=active_score,
            risk_category=active_cat,
            loan_offer=active_loan,
            is_simulated=is_slider_adjusted,
            baseline_score=base_score,
            baseline_limit=base_loan["limit"],
            baseline_features=baseline_features if is_slider_adjusted else None,
        )

    with tab_underwriter:
        render_underwriter_tab(
            persona=selected_persona,
            features=active_features,
            model=model,
            eval_metrics=eval_metrics,
            df_synthetic=df_synthetic,
            probability=active_prob,
            score=active_score,
            risk_category=active_cat,
        )

    # Footer
    st.markdown(
        "<div style='text-align: center; color: #94A3B8; font-size: 0.82rem; padding: 20px 0;'>"
        "AltScore AI • Hackathon Alternative Credit Scoring & Fairness Audit Prototype • Open Research Demo"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
