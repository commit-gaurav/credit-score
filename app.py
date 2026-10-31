"""AltScore AI — Explainable Alternative Credit Intelligence & Algorithmic Fairness Platform.

Flagship FinTech credit intelligence dashboard designed for thin-file borrower evaluation.
Featuring:
- Design A: The Consumer Trust Passbook (Warm, mobile-optimized, jargon-free)
- Design B: Dual-Mode Switcher (Borrower Lens vs. Institutional Underwriter & Auditor Lens)
- Design C: The Interactive What-If Coach (Real-time reactive scenario simulation)
"""

from pathlib import Path
import sys
import textwrap
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
# HTML / Markdown Rendering Helper (Prevents Indented Code Block Leakage)
# =============================================================================
def render_html(html_str: str) -> None:
    """Render HTML safely without indentation-induced Markdown code blocks.

    Strips all leading whitespace line-by-line so CommonMark never treats
    HTML lines as 4-space indented <pre><code> blocks.
    """
    cleaned = textwrap.dedent(html_str).strip()
    lines = [line.strip() for line in cleaned.splitlines()]
    clean_html = "\n".join(lines)
    st.markdown(clean_html, unsafe_allow_html=True)


# =============================================================================
# 1. Custom CSS Injection (Flagship FinTech Design System & Mobile First)
# =============================================================================
def inject_custom_css() -> None:
    """Inject responsive FinTech design system with CSS tokens and dark mode overrides."""
    css_content = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

    /* --------------------------------------------------
       CSS Design Tokens (:root)
    -------------------------------------------------- */
    :root {
        --bg-main: #F8FAFC;
        --text-primary: #0F172A;
        --text-secondary: #334155;
        --text-muted: #64748B;
        --card-bg: #FFFFFF;
        --card-border: #E2E8F0;
        --sidebar-bg: #1E293B;
        --sidebar-text: #F1F5F9;
        --accent-blue: #2563EB;
        --accent-blue-hover: #1D4ED8;
        --accent-emerald: #10B981;
        --accent-indigo: #4F46E5;
        --risk-low-bg: #ECFDF5;
        --risk-low-text: #065F46;
        --risk-low-border: #A7F3D0;
        --risk-med-bg: #FFFBEB;
        --risk-med-text: #92400E;
        --risk-med-border: #FDE68A;
        --risk-high-bg: #FEF2F2;
        --risk-high-text: #991B1B;
        --risk-high-border: #FECACA;
    }

    /* --------------------------------------------------
       Main App Canvas & Font Reset
    -------------------------------------------------- */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Constrained Container Layout */
    .main .block-container {
        max-width: 1200px !important;
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        color: var(--text-primary) !important;
    }

    /* High-Contrast Canvas Text Normalization */
    .main p, .main span, .main label, .main div,
    .main div[data-testid="stMarkdownContainer"] p,
    .main div[data-testid="stMarkdownContainer"] span,
    .main div[data-testid="stMarkdownContainer"] li {
        color: var(--text-primary) !important;
    }

    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* --------------------------------------------------
       Streamlit Boilerplate Cleanup (Keep Sidebar Toggle)
    -------------------------------------------------- */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .stDeployButton, [data-testid="stToolbarActions"], footer, #MainMenu {
        display: none !important;
    }
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        color: var(--text-primary) !important;
        background: #FFFFFF !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px !important;
        padding: 4px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }

    /* --------------------------------------------------
       Sidebar Styling (Deep Slate Dark Aesthetic)
    -------------------------------------------------- */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid #334155 !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--sidebar-text) !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: var(--sidebar-text) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #0F172A !important;
        border-color: #475569 !important;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSliderTickBar"] {
        background: #475569 !important;
    }

    /* --------------------------------------------------
       Hero Banner Card
    -------------------------------------------------- */
    .alt-hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: #FFFFFF !important;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.10);
    }
    .alt-hero-title {
        font-size: 1.95rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin: 0;
        letter-spacing: -0.025em;
    }
    .alt-hero-subtitle {
        font-size: 0.94rem;
        color: #94A3B8 !important;
        margin-top: 5px;
    }
    .alt-pill-badge {
        background: rgba(255, 255, 255, 0.10);
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.20);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.80rem;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.02em;
        backdrop-filter: blur(8px);
    }

    /* --------------------------------------------------
       Disclaimer Alert Banner (Soft Amber Theme)
    -------------------------------------------------- */
    .alt-disclaimer-banner {
        background-color: #FFFBEB !important;
        border: 1px solid #F59E0B !important;
        border-left: 5px solid #F59E0B !important;
        border-radius: 10px !important;
        padding: 14px 20px !important;
        margin-bottom: 18px !important;
        color: #92400E !important;
        font-size: 0.88rem !important;
        line-height: 1.5 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .alt-disclaimer-banner strong {
        color: #78350F !important;
    }

    /* --------------------------------------------------
       Segmented Lens Switcher (Fixes Invisible Text & Red Dot)
    -------------------------------------------------- */
    div[data-testid="stRadio"] {
        margin: 18px 0 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 8px !important;
        background: #E2E8F0 !important;
        padding: 6px !important;
        border-radius: 12px !important;
        width: fit-content !important;
        border: 1px solid #CBD5E1 !important;
    }
    div[data-testid="stRadio"] label {
        background: transparent !important;
        padding: 10px 22px !important;
        border-radius: 9px !important;
        cursor: pointer !important;
        margin: 0 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid transparent !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(255, 255, 255, 0.5) !important;
    }
    /* Unselected Tab Text */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label *,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        letter-spacing: -0.01em !important;
    }
    /* Active Selected Pill: FinTech Blue + Crisp White Text */
    div[data-testid="stRadio"] label:has(input:checked),
    div[data-testid="stRadio"] label[data-checked="true"] {
        background: #2563EB !important;
        box-shadow: 0 4px 8px -1px rgba(37, 99, 235, 0.35), 0 2px 4px -1px rgba(37, 99, 235, 0.2) !important;
        border: 1px solid #1D4ED8 !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) *,
    div[data-testid="stRadio"] label:has(input:checked) p,
    div[data-testid="stRadio"] label:has(input:checked) span,
    div[data-testid="stRadio"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stRadio"] label[data-checked="true"] *,
    div[data-testid="stRadio"] label[data-checked="true"] p,
    div[data-testid="stRadio"] label[data-checked="true"] span,
    div[data-testid="stRadio"] label[data-checked="true"] div[data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    /* Hide Default Radio Dot/Circle */
    div[data-testid="stRadio"] input[type="radio"],
    div[data-testid="stRadio"] div[data-testid="stRadioCircle"] {
        display: none !important;
    }

    /* --------------------------------------------------
       FinTech Content Cards & Micro-Interactions
    -------------------------------------------------- */
    .alt-card {
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 14px !important;
        padding: 22px 24px !important;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 18px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        overflow: hidden !important;
        word-wrap: break-word !important;
    }
    .alt-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.07) !important;
    }
    .alt-card-header {
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: var(--text-muted) !important;
        margin-bottom: 12px !important;
        border-bottom: 1px solid #F1F5F9 !important;
        padding-bottom: 8px !important;
    }

    /* Card Top Accent Borders */
    .card-accent-emerald {
        border-top: 4px solid var(--accent-emerald) !important;
    }
    .card-accent-blue {
        border-top: 4px solid var(--accent-blue) !important;
    }
    .card-accent-indigo {
        border-top: 4px solid var(--accent-indigo) !important;
    }

    /* Semantic Status Badges */
    .badge-low-risk {
        background-color: var(--risk-low-bg) !important;
        color: var(--risk-low-text) !important;
        border: 1px solid var(--risk-low-border) !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.84rem;
        display: inline-block;
    }
    .badge-med-risk {
        background-color: var(--risk-med-bg) !important;
        color: var(--risk-med-text) !important;
        border: 1px solid var(--risk-med-border) !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.84rem;
        display: inline-block;
    }
    .badge-high-risk {
        background-color: var(--risk-high-bg) !important;
        color: var(--risk-high-text) !important;
        border: 1px solid var(--risk-high-border) !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.84rem;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #F1F5F9 !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
        padding: 4px 11px;
        border-radius: 9999px;
        font-weight: 500;
        font-size: 0.80rem;
        display: inline-block;
    }

    /* Progress Meters */
    .meter-bg {
        background-color: #E2E8F0;
        border-radius: 8px;
        height: 12px;
        width: 100%;
        overflow: hidden;
        margin: 8px 0 10px 0;
    }
    .meter-fill-emerald {
        background-color: var(--accent-emerald);
        height: 100%;
        border-radius: 8px;
    }
    .meter-fill-amber {
        background-color: #F59E0B;
        height: 100%;
        border-radius: 8px;
    }
    .meter-fill-coral {
        background-color: #EF4444;
        height: 100%;
        border-radius: 8px;
    }

    /* Plain English Explainer Card */
    .explainer-box {
        background-color: #F8FAFC !important;
        border-left: 4px solid var(--accent-blue) !important;
        padding: 14px 18px !important;
        border-radius: 0 10px 10px 0 !important;
        margin-top: 12px !important;
        font-size: 0.88rem !important;
        color: #334155 !important;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }

    /* --------------------------------------------------
       Mobile Responsive Overrides (@media max-width: 768px)
    -------------------------------------------------- */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 0.75rem !important;
            padding-bottom: 1.75rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        .alt-hero-banner {
            padding: 18px 20px !important;
            border-radius: 12px !important;
        }
        .alt-hero-title {
            font-size: 1.55rem !important;
        }
        .alt-hero-subtitle {
            font-size: 0.84rem !important;
        }
        .alt-card {
            padding: 16px 18px !important;
            border-radius: 12px !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] {
            flex-direction: column !important;
            width: 100% !important;
        }
        div[data-testid="stRadio"] label {
            width: 100% !important;
            text-align: center !important;
            justify-content: center !important;
        }
    }
    </style>
    """
    render_html(css_content)


# =============================================================================
# Business Logic & Micro-Loan Computation
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
        return '<span class="badge-low-risk">● Low Risk (Approved)</span>'
    elif category == "Medium Risk":
        return '<span class="badge-med-risk">▲ Medium Risk (Manageable)</span>'
    else:
        return '<span class="badge-high-risk">■ High Risk (Needs Review)</span>'


# =============================================================================
# 2. Modular Component: Sidebar (Controls & What-If Coach)
# =============================================================================
def render_sidebar(personas: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float], bool]:
    """Render the sidebar with persona selector and interactive What-If sliders."""
    render_html(
        """
        <div style="padding-bottom: 6px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF;">AltScore Controls</div>
            <div style="font-size: 0.82rem; color: #94A3B8;">Applicant Selection & Scenario Coach</div>
        </div>
        """
    )
    st.sidebar.divider()

    # 1. Applicant Profile Selection
    st.sidebar.markdown("#### 👤 Select Applicant Profile")
    persona_keys = list(personas.keys())
    selected_key = st.sidebar.selectbox(
        label="Choose a thin-file borrower archetype:",
        options=persona_keys,
        format_func=lambda k: f"{personas[k]['name']} ({personas[k]['role'].split('(')[0].strip()})",
    )
    selected_persona = personas[selected_key]

    render_html(
        f"""
        <div style="background: #0F172A; padding: 12px 14px; border-radius: 10px; border: 1px solid #334155; margin-bottom: 12px; font-size: 0.82rem;">
            <div>• <strong>Role:</strong> {selected_persona['role']}</div>
            <div>• <strong>Location:</strong> {selected_persona['demographics']['geography']} ({selected_persona['demographics']['gender']})</div>
            <div>• <strong>Liquid Savings:</strong> ₹{selected_persona['liquid_balance']:,.2f}</div>
        </div>
        """
    )

    # 2. Extract Baseline Features
    baseline_features = extract_features(
        transactions=selected_persona["transactions"],
        liquid_balance=selected_persona["liquid_balance"],
    )

    # 3. Interactive What-If Coach Sliders (Design C)
    st.sidebar.divider()
    st.sidebar.markdown("#### 🎛️ Interactive What-If Coach")
    st.sidebar.caption("Adjust behavioral inputs to simulate real-time credit score and loan limit shifts:")

    adj_income = st.sidebar.slider(
        label="Income Flow Stability",
        min_value=0.0,
        max_value=1.0,
        value=float(baseline_features["income_stability"]),
        step=0.01,
        key=f"slider_inc_{selected_key}",
        help="Consistency of month-over-month earnings (1.0 = completely stable).",
    )

    adj_bills = st.sidebar.slider(
        label="Bill Payment Consistency",
        min_value=0.0,
        max_value=1.0,
        value=float(baseline_features["bill_payment_consistency"]),
        step=0.01,
        key=f"slider_bill_{selected_key}",
        help="Proportion of utility bills paid on time (1.0 = 100% on-time).",
    )

    adj_regularity = st.sidebar.slider(
        label="Payment Regularity",
        min_value=0.0,
        max_value=1.0,
        value=float(baseline_features["payment_regularity"]),
        step=0.01,
        key=f"slider_reg_{selected_key}",
        help="Proportion of months with recorded income transactions.",
    )

    adj_liquidity = st.sidebar.slider(
        label="Emergency Liquidity Buffer (Mos)",
        min_value=0.0,
        max_value=6.0,
        value=float(baseline_features["financial_liquidity_buffer"]),
        step=0.1,
        key=f"slider_liq_{selected_key}",
        help="Months of essential living expenses covered by savings (capped at 6.0).",
    )

    sim_features = {
        "income_stability": adj_income,
        "bill_payment_consistency": adj_bills,
        "payment_regularity": adj_regularity,
        "financial_liquidity_buffer": adj_liquidity,
    }

    is_slider_adjusted = sim_features != baseline_features

    return selected_persona, baseline_features, sim_features, is_slider_adjusted


# =============================================================================
# 3. Modular Component: Hero Banner
# =============================================================================
def render_hero_banner() -> None:
    """Render the Deep Navy Hero Banner with title and pill badge."""
    render_html(
        """
        <div class="alt-hero-banner">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h1 class="alt-hero-title">AltScore AI</h1>
                    <div class="alt-hero-subtitle">
                        Explainable Alternative Credit Intelligence & Algorithmic Fairness Platform
                    </div>
                </div>
                <div>
                    <span class="alt-pill-badge">
                        Synthetic Research Prototype • Non-Lending Tool
                    </span>
                </div>
            </div>
        </div>
        """
    )


# =============================================================================
# 4. Modular Component: Disclaimer & Summary KPIs
# =============================================================================
def render_disclaimer_and_kpis(
    probability: float,
    score: int,
    risk_category: str,
    loan_offer: Dict[str, Any],
    is_simulated: bool = False,
    baseline_score: int = 0,
    baseline_loan: Dict[str, Any] = None,
) -> None:
    """Render the soft amber disclaimer banner and top-level summary KPI cards."""
    # Amber Disclaimer Banner
    render_html(
        """
        <div class="alt-disclaimer-banner">
            <strong>⚠️ Mandatory Research Prototype Notice:</strong><br/>
            AltScore AI is an educational and research prototype. All borrower profiles, financial transactions,
            scores, and fairness metrics are entirely synthetic simulations. This tool does <strong>not</strong> access
            real consumer banking records, does not produce legal credit bureau scores (e.g. CIBIL/FICO), and must not be
            used for actual lending decisions.
        </div>
        """
    )

    # Top-Level KPI Summary Cards
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

    with col_kpi1:
        score_color = "#059669" if score >= 810 else ("#D97706" if score >= 720 else "#DC2626")
        score_delta_html = ""
        if is_simulated and baseline_score != 0:
            diff = score - baseline_score
            sign = "+" if diff > 0 else ""
            d_color = "#059669" if diff > 0 else ("#DC2626" if diff < 0 else "#64748B")
            score_delta_html = f'<span style="font-size: 0.90rem; font-weight: 700; color: {d_color}; margin-left: 6px; white-space: nowrap;">({sign}{diff} pts)</span>'

        render_html(
            f"""
            <div class="alt-card card-accent-emerald" style="height: 100%; padding: 18px 22px;">
                <div class="alt-card-header">Illustrative Credit Score</div>
                <div style="display: flex; align-items: baseline; gap: 4px; flex-wrap: nowrap; white-space: nowrap; margin: 4px 0;">
                    <span style="font-size: 2.3rem; font-weight: 800; color: {score_color}; line-height: 1;">{score}</span>
                    <span style="font-size: 1.05rem; color: #64748B; font-weight: 600; white-space: nowrap;">/ 900</span>
                    {score_delta_html}
                </div>
                <div style="margin-top: 10px;">{get_risk_badge_html(risk_category)}</div>
            </div>
            """
        )

    with col_kpi2:
        render_html(
            f"""
            <div class="alt-card card-accent-blue" style="height: 100%; padding: 18px 22px;">
                <div class="alt-card-header">Repayment Probability</div>
                <div style="font-size: 2.3rem; font-weight: 800; color: #0F172A; line-height: 1; margin: 4px 0;">
                    {probability:.1%}
                </div>
                <div style="margin-top: 10px;">
                    <span class="badge-neutral">Calibrated Logistic Sigmoid</span>
                </div>
            </div>
            """
        )

    with col_kpi3:
        limit_delta_html = ""
        if is_simulated and baseline_loan:
            l_diff = loan_offer["limit"] - baseline_loan["limit"]
            if l_diff != 0:
                l_sign = "+" if l_diff > 0 else ""
                l_col = "#059669" if l_diff > 0 else "#DC2626"
                limit_delta_html = f'<span style="font-size: 0.88rem; font-weight: 700; color: {l_col}; margin-left: 6px; white-space: nowrap;">({l_sign}₹{abs(l_diff):,})</span>'

        render_html(
            f"""
            <div class="alt-card card-accent-indigo" style="height: 100%; padding: 18px 22px;">
                <div class="alt-card-header">Pre-Approved Micro-Line</div>
                <div style="display: flex; align-items: baseline; margin: 4px 0; flex-wrap: nowrap; white-space: nowrap;">
                    <span style="font-size: 2.3rem; font-weight: 800; color: #0F172A; line-height: 1;">{loan_offer['formatted_limit']}</span>
                    {limit_delta_html}
                </div>
                <div style="margin-top: 10px;">
                    <span class="{loan_offer['badge_class']}">{loan_offer['badge_text']}</span>
                </div>
            </div>
            """
        )


# =============================================================================
# 5. Modular Component: The Digital Passbook (Design A - Borrower Lens)
# =============================================================================
def render_digital_passbook(
    persona: Dict[str, Any],
    features: Dict[str, float],
    score: int,
    risk_category: str,
    loan_offer: Dict[str, Any],
    is_simulated: bool = False,
    baseline_score: int = 0,
    baseline_limit: int = 0,
) -> None:
    """Render Design A: The Consumer-Friendly Trust Passbook."""
    # Profile & Identity Overview Card
    render_html(
        f"""
        <div class="alt-card">
            <div class="alt-card-header">Applicant Digital Passbook</div>
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <h2 style="margin: 0; color: #0F172A; font-size: 1.6rem; font-weight: 800;">{persona['name']}</h2>
                    <span style="color: #475569; font-size: 0.95rem; font-weight: 500;">{persona['role']}</span>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span class="badge-neutral">Region: {persona['demographics']['geography']}</span>
                    <span class="badge-neutral">Savings: ₹{persona['liquid_balance']:,.2f}</span>
                </div>
            </div>
            <p style="margin-top: 12px; color: #334155; font-size: 0.92rem; line-height: 1.5; margin-bottom: 0;">
                {persona['summary']}
            </p>
        </div>
        """
    )

    # Credit Health & Micro-Loan Offer Breakdown
    col_score_card, col_offer_card = st.columns([1.1, 1.3])

    with col_score_card:
        score_color = "#059669" if score >= 810 else ("#D97706" if score >= 720 else "#DC2626")
        progress_pct = max(0, min(100, int(((score - 300) / 600) * 100)))
        fill_class = "meter-fill-emerald" if score >= 810 else ("meter-fill-amber" if score >= 720 else "meter-fill-coral")

        render_html(
            f"""
            <div class="alt-card" style="height: 100%;">
                <div class="alt-card-header">Credit Standing Gauge</div>
                <div style="display: flex; align-items: baseline; gap: 6px; flex-wrap: nowrap; white-space: nowrap;">
                    <span style="font-size: 3.2rem; font-weight: 800; color: {score_color}; line-height: 1;">{score}</span>
                    <span style="font-size: 1.1rem; color: #64748B; font-weight: 600;">/ 900</span>
                </div>
                <div style="margin: 10px 0;">{get_risk_badge_html(risk_category)}</div>
                <div class="meter-bg">
                    <div class="{fill_class}" style="width: {progress_pct}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #94A3B8; font-weight: 600;">
                    <span>300 (Needs Work)</span>
                    <span>720 (Fair)</span>
                    <span>900 (Excellent)</span>
                </div>
            </div>
            """
        )

    with col_offer_card:
        render_html(
            f"""
            <div class="alt-card" style="height: 100%;">
                <div class="alt-card-header">Instant Micro-Line Terms</div>
                <div style="display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;">
                    <span style="font-size: 2.3rem; font-weight: 800; color: #0F172A; line-height: 1;">{loan_offer['formatted_limit']}</span>
                    <span class="{loan_offer['badge_class']}">{loan_offer['badge_text']}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px;">
                    <div style="background: #F8FAFC; padding: 10px 14px; border-radius: 8px; border: 1px solid #E2E8F0;">
                        <span style="font-size: 0.76rem; color: #64748B; font-weight: 700; text-transform: uppercase;">ESTIMATED INTEREST</span>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{loan_offer['interest_rate']}</div>
                    </div>
                    <div style="background: #F8FAFC; padding: 10px 14px; border-radius: 8px; border: 1px solid #E2E8F0;">
                        <span style="font-size: 0.76rem; color: #64748B; font-weight: 700; text-transform: uppercase;">REPAYMENT TENURE</span>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{loan_offer['tenure']}</div>
                    </div>
                </div>
                <p style="font-size: 0.82rem; color: #64748B; margin-top: 10px; margin-bottom: 0;">
                    Disburses instantly via UPI directly to registered bank account upon confirmation.
                </p>
            </div>
            """
        )

    # Daily Financial Health Habits (4 Behavioral Signals)
    st.markdown("### Daily Financial Health Habits")
    st.caption("How daily cash-flow habits contribute to your credit health without credit cards:")

    m1, m2 = st.columns(2)
    with m1:
        inc_pct = int(features["income_stability"] * 100)
        inc_fill = "meter-fill-emerald" if inc_pct >= 80 else ("meter-fill-amber" if inc_pct >= 60 else "meter-fill-coral")
        render_html(
            f"""
            <div class="alt-card">
                <div style="display: flex; justify-content: space-between; font-weight: 600;">
                    <span style="color: #0F172A;">Income Flow Reliability</span>
                    <span style="color: #059669; font-weight: 700;">{inc_pct}% Score</span>
                </div>
                <div class="meter-bg">
                    <div class="{inc_fill}" style="width: {inc_pct}%;"></div>
                </div>
                <div style="font-size: 0.84rem; color: #475569;">
                    Evaluates consistency of monthly cash inflows. Stable gig or shop income builds trust.
                </div>
            </div>
            """
        )

        bill_pct = int(features["bill_payment_consistency"] * 100)
        bill_fill = "meter-fill-emerald" if bill_pct >= 90 else ("meter-fill-amber" if bill_pct >= 70 else "meter-fill-coral")
        render_html(
            f"""
            <div class="alt-card">
                <div style="display: flex; justify-content: space-between; font-weight: 600;">
                    <span style="color: #0F172A;">On-Time Bill Discipline</span>
                    <span style="color: #059669; font-weight: 700;">{bill_pct}% On-Time</span>
                </div>
                <div class="meter-bg">
                    <div class="{bill_fill}" style="width: {bill_pct}%;"></div>
                </div>
                <div style="font-size: 0.84rem; color: #475569;">
                    Tracks timely settlement of utility, mobile recharge, and rent payments.
                </div>
            </div>
            """
        )

    with m2:
        reg_pct = int(features["payment_regularity"] * 100)
        reg_fill = "meter-fill-emerald" if reg_pct >= 80 else ("meter-fill-amber" if reg_pct >= 60 else "meter-fill-coral")
        render_html(
            f"""
            <div class="alt-card">
                <div style="display: flex; justify-content: space-between; font-weight: 600;">
                    <span style="color: #0F172A;">Activity Regularity</span>
                    <span style="color: #059669; font-weight: 700;">{reg_pct}% Active</span>
                </div>
                <div class="meter-bg">
                    <div class="{reg_fill}" style="width: {reg_pct}%;"></div>
                </div>
                <div style="font-size: 0.84rem; color: #475569;">
                    Checks for regular cash inflows each month, proving active trade or gig employment.
                </div>
            </div>
            """
        )

        buffer_val = features["financial_liquidity_buffer"]
        buf_pct = min(100, int((buffer_val / 6.0) * 100))
        buf_fill = "meter-fill-emerald" if buffer_val >= 3.0 else ("meter-fill-amber" if buffer_val >= 1.5 else "meter-fill-coral")
        render_html(
            f"""
            <div class="alt-card">
                <div style="display: flex; justify-content: space-between; font-weight: 600;">
                    <span style="color: #0F172A;">Emergency Savings Cushion</span>
                    <span style="color: #059669; font-weight: 700;">{buffer_val:.1f} Months</span>
                </div>
                <div class="meter-bg">
                    <div class="{buf_fill}" style="width: {buf_pct}%;"></div>
                </div>
                <div style="font-size: 0.84rem; color: #475569;">
                    Number of months savings can sustain living expenses (capped at 6.0 months).
                </div>
            </div>
            """
        )

    # Actionable Tips Card
    with st.expander("💡 Actionable Tips to Unlock Higher Credit Limits"):
        st.markdown(
            """
            - **Maintain a 2-Month Savings Buffer:** Storing even ₹5,000 more in your savings account significantly boosts your score.
            - **Never Delay Utility Bills:** Paying utility and phone bills on or before the due date keeps your bill discipline score at 100%.
            - **Deposit Cash Digitally:** Depositing cash collections into your UPI account weekly proves steady income.
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


# =============================================================================
# 6. Modular Component: Bank Underwriter & Risk Auditor View (Design B)
# =============================================================================
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
    """Render Design B: The Professional Risk Management and Audit Console."""
    render_html(
        """
        <div style="background: #0F172A; color: #F8FAFC; padding: 18px 22px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #334155;">
            <div style="font-size: 0.80rem; text-transform: uppercase; color: #38BDF8; font-weight: 700; letter-spacing: 0.05em;">
                Institutional Risk Underwriting Console
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; margin-top: 2px;">
                Algorithmic Risk Assessment & Demographic Compliance Audit
            </div>
        </div>
        """
    )

    # 1. Underwriting Decision Cards
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

    # 2. Local Linear Log-Odds Decomposition
    st.markdown("### 1. Local Prediction Explanation (Linear Decomposition)")
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

    render_html(
        """
        <div class="explainer-box">
            <strong>Plain English Explanation for Auditors:</strong><br/>
            - <strong>Model Weight (β):</strong> How sensitive the model is to each factor in general across training rows.<br/>
            - <strong>Local Contribution (β × x):</strong> The exact log-odds shift this applicant receives. A positive number pushes the decision toward approval; a negative number reduces repayment odds.<br/>
            - <strong>Zero Demographic Leakage:</strong> Gender and geography columns were completely excluded from the model feature matrix.
        </div>
        """
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
        render_html(
            f"""
            <div class="alt-card">
                <div class="alt-card-header">Dimension 1: Gender Audit</div>
                <div>• Reference Group (Higher Rate): <strong>{g_info['reference_group']} ({g_info['reference_rate']:.1%})</strong></div>
                <div>• Comparison Group (Lower Rate): <strong>{g_info['comparison_group']} ({g_info['comparison_rate']:.1%})</strong></div>
                <div style="margin-top: 14px;">
                    <span style="font-size: 0.82rem; color: #64748B; font-weight: 600;">DISPARATE IMPACT RATIO:</span>
                    <div style="font-size: 1.85rem; font-weight: 700; color: #0F172A;">{g_info['ratio_display']}</div>
                </div>
            </div>
            """
        )

    with col_geo:
        geo_info = audit_results["dimensions"]["geography"]
        render_html(
            f"""
            <div class="alt-card">
                <div class="alt-card-header">Dimension 2: Geography Audit</div>
                <div>• Reference Group (Higher Rate): <strong>{geo_info['reference_group']} ({geo_info['reference_rate']:.1%})</strong></div>
                <div>• Comparison Group (Lower Rate): <strong>{geo_info['comparison_group']} ({geo_info['comparison_rate']:.1%})</strong></div>
                <div style="margin-top: 14px;">
                    <span style="font-size: 0.82rem; color: #64748B; font-weight: 600;">DISPARATE IMPACT RATIO:</span>
                    <div style="font-size: 1.85rem; font-weight: 700; color: #0F172A;">{geo_info['ratio_display']}</div>
                </div>
            </div>
            """
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


# =============================================================================
# 7. Modular Component: What-If Comparison Card
# =============================================================================
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
        render_html(
            f"""
            <div class="alt-card">
                <div class="alt-card-header">Baseline Standing ({persona_name})</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #1E293B;">{baseline_score} <span style="font-size: 1rem; color: #64748B;">/ 900</span></div>
                <div style="margin: 6px 0;">{b_badge}</div>
                <div style="margin-top: 10px; font-size: 0.90rem;">
                    <div>• Repayment Probability: <strong>{baseline_prob:.1%}</strong></div>
                    <div>• Eligible Credit Line: <strong>{baseline_loan['formatted_limit']}</strong></div>
                </div>
            </div>
            """
        )

    with c_sim:
        score_diff = sim_score - baseline_score
        limit_diff = sim_loan["limit"] - baseline_loan["limit"]
        diff_color = "#059669" if score_diff > 0 else ("#DC2626" if score_diff < 0 else "#64748B")
        diff_text = f"+{score_diff} pts" if score_diff > 0 else (f"{score_diff} pts" if score_diff < 0 else "No change")
        limit_diff_text = f"+₹{limit_diff:,}" if limit_diff > 0 else (f"-₹{abs(limit_diff):,}" if limit_diff < 0 else "No change")
        s_badge = get_risk_badge_html(sim_category)

        render_html(
            f"""
            <div class="alt-card" style="border: 2px solid #2563EB;">
                <div class="alt-card-header" style="color: #1D4ED8;">Simulated Outcome (Adjusted Inputs)</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #1E293B;">
                    {sim_score} <span style="font-size: 1rem; color: {diff_color}; font-weight: 700;">({diff_text})</span>
                </div>
                <div style="margin: 6px 0;">{s_badge}</div>
                <div style="margin-top: 10px; font-size: 0.90rem;">
                    <div>• Repayment Probability: <strong>{sim_prob:.1%}</strong> ({sim_prob - baseline_prob:+.1%})</div>
                    <div>• Eligible Credit Line: <strong>{sim_loan['formatted_limit']}</strong> ({limit_diff_text})</div>
                </div>
            </div>
            """
        )


# =============================================================================
# Main Application Entrypoint
# =============================================================================
def main() -> None:
    """Build and render the unified AltScore AI FinTech dashboard."""
    st.set_page_config(
        page_title="AltScore AI — Alternative Credit & Fairness Platform",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 1. Inject Responsive FinTech CSS
    inject_custom_css()

    # 2. Render Hero Header Banner
    render_hero_banner()

    # 3. Load In-Memory Data & Train Baseline Model
    personas = get_personas()
    df_synthetic = generate_synthetic_dataset(seed=DEFAULT_SEED, row_count=DEFAULT_ROW_COUNT)
    model, eval_metrics = train_model(df_synthetic, seed=DEFAULT_SEED)

    # 4. Render Modular Sidebar (Applicant Selector + What-If Coach)
    selected_persona, baseline_features, sim_features, is_slider_adjusted = render_sidebar(personas)

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

    # 7. Render Soft Amber Disclaimer & Summary KPI Cards
    render_disclaimer_and_kpis(
        probability=active_prob,
        score=active_score,
        risk_category=active_cat,
        loan_offer=active_loan,
        is_simulated=is_slider_adjusted,
        baseline_score=base_score,
        baseline_loan=base_loan,
    )

    # 8. Persistent Segmented Lens Switcher (Borrower vs Underwriter)
    view_mode = st.radio(
        label="Operational Lens Switcher",
        options=[
            "Consumer Trust Passbook (Borrower Lens)",
            "Institutional Underwriting & Fairness Audit (Bank Lens)",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    # 9. Render Selected Operational View
    if view_mode == "Consumer Trust Passbook (Borrower Lens)":
        render_digital_passbook(
            persona=selected_persona,
            features=active_features,
            score=active_score,
            risk_category=active_cat,
            loan_offer=active_loan,
            is_simulated=is_slider_adjusted,
            baseline_score=base_score,
            baseline_limit=base_loan["limit"],
        )

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
    render_html(
        """
        <div style="text-align: center; color: #64748B; font-size: 0.82rem; padding: 10px 0;">
            AltScore AI • Hackathon Alternative Credit Scoring & Fairness Audit Prototype • Open Research Demo
        </div>
        """
    )


if __name__ == "__main__":
    main()
