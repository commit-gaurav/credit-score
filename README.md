# AltScore AI: Explainable Alternative Credit Scoring & Fairness Audit

AltScore AI is an explainable alternative credit intelligence prototype designed for hackathon demonstration. It evaluates borrowers who lack formal credit bureau histories—such as gig workers, small shopkeepers, rural artisans, and first-time borrowers—using cash-flow behavioral signals, transparent machine learning, interactive scenario exploration, and demographic fairness auditing.

---

> ### ⚠️ Mandatory Prototype & Ethical Disclaimer
> **Synthetic Demonstration Only:** AltScore AI is an educational and research prototype. All borrower personas, financial transactions, training datasets, risk tiers, illustrative scores, and fairness metrics are entirely synthetic simulations. This tool is **not** an established credit bureau scoring engine (such as CIBIL, Experian, or FICO), does not access real consumer banking records, and **must not** be used for actual credit underwriting or lending decisions.

---

## 1. Quick Setup & Launch Instructions

Follow these steps to launch the dashboard locally on your machine:

### Prerequisites
* Python 3.10 or higher installed.
* PowerShell, Command Prompt (Windows), or Bash (macOS/Linux).

### Step-by-Step Launch Commands

```powershell
# 1. Clone or navigate to the project directory
cd "d:\credit project"

# 2. Create a virtual environment (if not already present)
python -m venv .venv

# 3. Activate the virtual environment
# On Windows PowerShell:
.\.venv\bin\Activate.ps1
# (Or on Windows Command Prompt: .\.venv\Scripts\activate.bat)
# (Or on macOS / Linux: source .venv/bin/activate)

# 4. Install required dependencies
pip install -r requirements.txt

# 5. Launch the Streamlit dashboard
streamlit run app.py
```

*Direct PowerShell fallback (runs directly without environment activation):*
```powershell
.\.venv\bin\python.exe -m streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 2. Project Architecture & File Organization

The application is structured into focused, modular packages. Data and model objects remain entirely in memory; no external databases or network services are required.

```
credit-score/
├── app.py                      # Interactive Streamlit dashboard and user presentation flow
├── requirements.txt            # Minimal Python dependencies (Streamlit, pandas, scikit-learn)
├── PLAN.md                     # Architecture specification, 10-slice build order, decisions log
├── README.md                   # Setup instructions, demo script, and technical explanations
├── BACKEND.md                  # Detailed plain-English backend workflow guide
├── data/
│   ├── __init__.py             # Package initializer
│   ├── personas.py             # 3 synthetic profiles & 6-month transaction ledgers
│   └── synthetic_data.py       # Reproducible N=1,000 dataset generator (seed=42)
├── features/
│   ├── __init__.py             # Package initializer
│   └── extraction.py           # Feature extraction engine (4 behavioral signals)
├── model/
│   ├── __init__.py             # Package initializer
│   └── scoring.py              # Logistic regression, score mapping & linear explanations
└── fairness/
    ├── __init__.py             # Package initializer
    └── audit.py                # Demographic approval rates & disparate impact ratios
```

---

## 3. Data Flow & Technical Explanations

### How the Data Flows Through the System
1. **Persona Selection:** The user selects an applicant profile in the UI.
2. **Feature Extraction:** Raw transactions pass through `extract_features()`. Four normalized financial signals are computed. Demographics remain strictly isolated metadata.
3. **Synthetic Training Population:** An in-memory population ($N=1,000$) is generated with a fixed seed (`seed=42`) using a latent logistic formula with stochastic noise ($\epsilon$).
4. **Baseline Model Training:** A Scikit-Learn `LogisticRegression` classifier trains on 75% of the synthetic data ($N=750$) and evaluates on a 25% held-out test split ($N=250$).
5. **Prediction & Scoring:** The selected persona's features pass into the model. The repayment probability is mapped to an illustrative 300–900 score and risk category.
6. **Local Explanation:** The model's log-odds are decomposed into exact feature-level contributions ($\beta_i \times x_i$).
7. **What-If Simulation:** Users adjust feature sliders; modified values run through the exact same model pipeline to recompute probability, score, and explanation side-by-side with baseline.
8. **Fairness Audit:** Post-scoring demographic grouping calculates group approval rates and disparate impact ratios across Gender and Urban/Rural geography.

---

### The Four Behavioral Features (Formulas & Edge Cases)

| Feature | Range / Unit | Mathematical Formula | Explicit Missing-Data Policy |
| :--- | :---: | :--- | :--- |
| **Income Stability** | `0.00 – 1.00` | $\max\left(0, 1 - \frac{\sigma_{\text{inflow}}}{\mu_{\text{inflow}}}\right)$ across monthly inflows | If zero income is recorded, returns `0.00`. If only 1 month exists, variance is 0, returns `1.00`. |
| **Bill-Payment Consistency** | `0.00 – 1.00` | $\frac{\text{Count of On-Time Bills}}{\text{Total Scheduled Bills}}$ | **Zero bills recorded does NOT count as missed payments.** Default is neutral `1.00` (no adverse records). |
| **Payment Regularity** | `0.00 – 1.00` | $\frac{\text{Months With Inflows}}{\text{Total Observed Months}}$ | If no transactions exist, returns `0.00`. |
| **Liquidity Buffer** | `0.00 – 6.00 mos` | $\min\left(6.0, \frac{\text{Liquid Balance}}{\text{Avg Monthly Outflow}}\right)$ | If monthly outflow is zero, capped at `6.00` months. Negative balance floored at `0.00`. Strictly capped at `6.00` months. |

---

### The Scoring Model & Evaluation Metrics

* **Why Logistic Regression?**
  - **Explainability:** Produces direct, transparent linear weights ($\beta$) that can be fully explained to borrowers, eliminating "black-box" mystery.
  - **Calibration:** Outputs mathematically sound probabilities ($0.0$ to $1.0$).
  - **Speed:** Trains in milliseconds without requiring GPUs or external cloud services.
* **Held-Out Test Metrics ($N=250$ Synthetic Samples):**
  - **ROC-AUC:** `0.5834` (Measures test sample ranking quality).
  - **Accuracy:** `80.00%`
  - **Precision:** `80.80%`
  - **Recall:** `98.50%`
  - **F1 Score:** `0.8879`
  *(All metrics are computed on synthetic test distributions and labeled as synthetic results).*

---

### Illustrative Score & Risk Classification

* **Score Mapping Formula:**
  $$\text{Illustrative Score} = \text{clamp}\Big(300, 900, \text{round}(300 + 600 \times P_{\text{repayment}})\Big)$$
  *Note: This is an illustrative linear transformation, not an established credit bureau score (such as CIBIL or FICO).*

* **Documented Risk Categories:**
  - **Low Risk:** Repayment Probability $P \ge 85\%$ (Score $810 – 900$).
  - **Medium Risk:** Repayment Probability $70\% \le P < 85\%$ (Score $720 – 809$).
  - **High Risk:** Repayment Probability $P < 70\%$ (Score $300 – 719$).

---

### Demographic Fairness Audit & Disparate Impact

* **Documented Approval Threshold:** Borrowers with estimated repayment probability $P \ge 0.70$ (Score $\ge 720$) are approved.
* **Four-Fifths Rule (EEOC Disparate Impact Standard):**
  1. **Reference Group:** The demographic group with the higher approval rate.
  2. **Comparison Group:** The demographic group with the lower approval rate.
  3. **Disparate Impact Ratio:**
     $$\text{Disparate Impact Ratio} = \frac{\text{Approval Rate}_{\text{comparison}}}{\text{Approval Rate}_{\text{reference}}}$$
  4. If the ratio is $\ge 0.80$, it satisfies the Four-Fifths benchmark under synthetic test conditions.
* **Zero-Rate Protection:** If the reference group approval rate is $0.0\%$, the ratio is explicitly reported as `Undefined` rather than raising a division-by-zero error.
* **Synthetic Audit Results ($N=1,000$ Rows):**
  - **Gender Audit:** Female reference rate ($88.22\%$) vs. Male comparison rate ($87.79\%$) $\to$ **Ratio: $0.9951$** (Pass).
  - **Geography Audit:** Rural reference rate ($88.59\%$) vs. Urban comparison rate ($87.52\%$) $\to$ **Ratio: $0.9879$** (Pass).

---

## 4. Judge Demonstration Script (5-Minute Walkthrough)

Use this structured walkthrough when presenting AltScore AI to evaluators or judges:

```
[Demo Flow Overview]
1. Introduction & Disclaimer  -->  2. Three Personas Walkthrough  -->  3. Feature Extraction
4. Model & Held-Out Metrics   -->  5. Score & Risk Categorization -->  6. Local Explanations
7. Interactive What-If Sim    -->  8. Demographic Fairness Audit  -->  9. Governance Conclusion
```

### Step 1: The Problem & Prototype Disclaimer (30 Seconds)
- Open the dashboard at `http://localhost:8501`.
- State the core challenge: Traditional credit bureaus fail thin-file borrowers (gig workers, informal micro-entrepreneurs) who lack formal bank statements.
- Point to the **Prototype Disclaimer** banner: All data, scores, and metrics are synthetic learning demonstrations.

### Step 2: Persona Comparison (1.5 Minutes)
Use the dropdown in **Section 1** to compare the three contrasting archetypes:
1. **Ramesh Kumar (Rural Male - Kirana Store Owner):**
   - Steady daily grocery sales, $100\%$ on-time utility bills, $3.8$-month liquid buffer.
   - *Result:* Repayment Probability **$94.39\%$**, Illustrative Score **$866$**, **Low Risk**.
2. **Sunita Devi (Urban Female - Artisan Tailor):**
   - Seasonal wedding/festival income, $100\%$ on-time bills, moderate $2.1$-month buffer.
   - *Result:* Repayment Probability **$74.52\%$**, Illustrative Score **$747$**, **Medium Risk**.
3. **Aarav Sharma (Urban Male - Gig Delivery Courier):**
   - Volatile weekly earnings, $1$ missed bill payment, narrow $0.8$-month buffer.
   - *Result:* Repayment Probability **$59.39\%$**, Illustrative Score **$656$**, **High Risk**.

### Step 3: Model Explanation & Transparency (1 Minute)
- Scroll to **Section 7: Prediction Explanation**.
- Explain the difference between **global feature importance** ($\beta$) and **local prediction contribution** ($\beta \times x$).
- Show how the logit is decomposed linearly into positive and negative drivers:
  - For Ramesh: Strong positive contributions from income stability and liquidity buffer.
  - For Aarav: Lower bill consistency and a narrow buffer pull down his odds.
- Highlight the governance rule: statistical associations reflect synthetic patterns, not causal guarantees.

### Step 4: Interactive What-If Scenario Exploration (1 Minute)
- Scroll to **Section 8: What-If Simulator** with **Aarav Sharma** selected.
- Show how AltScore AI acts as a financial coaching tool rather than a black-box denial.
- Drag Aarav's **Liquidity Buffer** slider from $0.8 \to 3.5$ months and **Bill-Payment Consistency** from $0.67 \to 1.00$.
- Highlight the real-time update beside the baseline:
  - Estimated repayment probability rises from $59.4\% \to \approx 78\%$.
  - Illustrative score climbs from $656 \to \approx 770$.
  - Risk classification improves from **High Risk** to **Medium Risk**.
  - The feature contribution table updates dynamically to show the exact logit shift.

### Step 5: Fairness Audit & The Proxy Bias Dilemma (1 Minute)
- Scroll to **Section 9: Fairness & Disparate Impact Audit**.
- Show that both Gender ($0.9951$) and Geography ($0.9879$) pass the Four-Fifths rule.
- Conclude with the **Proxy Bias governance lesson**:
  - *"Excluding demographic columns from model inputs eliminates direct disparate treatment, but it does NOT eliminate indirect proxy bias. Neutral financial signals (like liquidity reserves) can still correlate with historical systemic inequalities. Responsible AI requires continuous post-scoring demographic auditing."*

---

## 5. Critical Prototype Limitations & Governance

1. **Wholly Synthetic Data:** No real consumer transactions or banking records are accessed. Metrics cannot be used to predict real-world default rates.
2. **Not a Real Credit Bureau Score:** The 300–900 score is an illustrative mathematical scaling of model probability ($300 + 600 \times P$). It carries no legal or regulatory credit bureau standing.
3. **Correlation vs. Causation:** Model feature weights reflect statistical correlations in synthetic data, not causal guarantees. Improving a borrower's liquidity buffer slider increases model odds, but real-world repayment depends on broader macro-economic factors.
4. **The Proxy Bias Dilemma:** Eliminating protected attributes (gender, location) from model inputs does not prevent indirect discrimination if financial features act as proxies for historical economic disparities.
5. **No Regulatory Compliance Claim:** These synthetic audit numbers do not establish statutory fair lending compliance under the Equal Credit Opportunity Act (ECOA) or FCRA.

---

## 6. License & Attribution

Released under the **MIT License**. Designed for hackathon presentations, open technical research in AI explainability, and algorithmic fairness demonstrations.
