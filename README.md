# AltScore AI: Explainable Alternative Credit Scoring & Fairness Audit

AltScore AI is an interactive, explainable alternative credit intelligence prototype designed to evaluate borrowers without traditional credit bureau histories (e.g., gig workers, informal artisans, and micro-entrepreneurs). Built with Python, Streamlit, pandas, and scikit-learn, the application demonstrates how alternative cash-flow behavioral signals can be extracted, modeled transparently, explained at the local decision level, simulated interactively, and audited for demographic fairness.

---

> ### ⚠️ Mandatory Ethical & Prototype Disclaimer
> **Synthetic Demonstration Only:** AltScore AI is an educational, research, and technical prototype. All borrower personas, transaction records, training datasets, scores, and fairness metrics are entirely synthetic simulations. This tool is **not** an established credit bureau scoring engine (such as CIBIL, Experian, or FICO), does not access real consumer financial data, and **must not** be used to make actual underwriting, lending, or credit denial decisions.

---

## 1. System Architecture

The codebase follows a modular design with separation between synthetic profiles, behavioral feature extraction, training data generation, explainable ML scoring, and demographic auditing:

```
credit-score/
├── app.py                      # Interactive Streamlit dashboard and UI rendering
├── requirements.txt            # Python dependencies (Streamlit, pandas, scikit-learn, etc.)
├── PLAN.md                     # Architecture specification, 10-slice build plan, decisions log
├── README.md                   # Complete documentation, math formulations, and judge demo script
├── BACKEND.md                  # Plain-English backend architecture and workflow guide
├── data/
│   ├── __init__.py
│   ├── personas.py             # 3 synthetic profiles & 6-month transaction ledgers
│   └── synthetic_data.py       # Reproducible N=1,000 dataset generator (seed=42)
├── features/
│   ├── __init__.py
│   └── extraction.py           # Cash-flow behavioral feature extraction engine
├── model/
│   ├── __init__.py
│   └── scoring.py              # Logistic regression, scoring, calibration & local explanations
└── fairness/
    ├── __init__.py
    └── audit.py                # Disparate impact auditing across gender & geography
```

> 📖 **Deep Dive into the Backend:** For an end-to-end, plain-English breakdown of how every file and library works in sequence, read [BACKEND.md](file:///d:/credit%20project/BACKEND.md).

---

## 2. Core Feature Formulations & Logic

To evaluate thin-file and credit-invisible individuals fairly, the model evaluates four behavioral financial features extracted from monthly cash flows. Demographic fields (gender, age, location) are strictly excluded from the feature vector $X$.

### 1. Income Stability
* **Definition:** Consistency of monthly income inflows over the observation window.
* **Formula:** $\text{Income Stability} = \max\left(0.0, 1.0 - \frac{\sigma_{\text{inflow}}}{\mu_{\text{inflow}}}\right)$
* **Units:** Dimensionless index bounded in $[0.0, 1.0]$. A value of $1.0$ indicates perfectly steady income; higher coefficient of variation ($\sigma / \mu$) lowers stability.
* **Edge Cases:** If $\mu_{\text{inflow}} \le 0$, stability is assigned $0.0$.

### 2. Bill-Payment Consistency
* **Definition:** Proportion of utility, rent, and recurring commitments paid on or before the due date.
* **Formula:** $\text{Bill Consistency} = \frac{\text{Count of On-Time Bill Payments}}{\text{Total Count of Scheduled Bills}}$
* **Units:** Proportion bounded in $[0.0, 1.0]$.
* **Missing-Data Handling:** If no scheduled bills exist, defaults to $1.0$ (neutral/unpenalized). Unpaid or missing transactions are explicitly distinguished and never silently counted as on-time.

### 3. Payment Regularity
* **Definition:** Uniformity of cash outflows across the month, penalizing lump-sum exhaustion.
* **Formula:** $\text{Payment Regularity} = \max\left(0.0, 1.0 - \frac{\sigma_{\text{weekly\_spend}}}{\mu_{\text{weekly\_spend}}}\right)$
* **Units:** Dimensionless index bounded in $[0.0, 1.0]$. Smooth weekly distributions approach $1.0$.

### 4. Financial Liquidity Buffer
* **Definition:** Number of months the borrower can sustain average living expenses using current liquid reserves.
* **Formula:** $\text{Liquidity Buffer} = \min\left(6.0, \frac{\text{Current Liquid Balance}}{\text{Average Monthly Outflows}}\right)$
* **Units:** Months of living expenses (capped strictly at $6.0$ months to prevent ultra-wealthy outliers from skewing linear risk estimates).
* **Missing-Data Handling:** If monthly outflows equal $0$, the buffer is capped at $6.0$ months.

---

## 3. Modeling, Calibration & Local Explanations

### Baseline Logistic Regression
* **Dataset:** $N = 1,000$ synthetic borrower records generated with a fixed seed (`DEFAULT_SEED = 42`).
* **Train / Held-Out Test Split:** Stratified 75% train ($N=750$) and 25% test ($N=250$).
* **Features:** Exclusively the 4 behavioral features above. Demographic columns (`gender`, `geography`, `age`) and target labels are omitted from training inputs.
* **Evaluation Metrics (Held-Out Test Set):**
  - **ROC-AUC:** $0.5834$
  - **Accuracy:** $80.00\%$
  - **Precision:** $80.80\%$
  - **Recall:** $98.50\%$
  - **F1 Score:** $0.8879$
  *(All metrics are labeled as synthetic demonstration metrics).*

### Score Mapping Formula
Estimated repayment probability $P \in [0.0, 1.0]$ is translated to an illustrative score:
$$\text{Illustrative Score} = \text{clamp}\Big(300, 900, \text{round}(300 + 600 \times P)\Big)$$
* **Note:** This represents an illustrative linear mapping, **not** an established bureau score (such as CIBIL or FICO).

### Documented Risk Categorization
* **Low Risk:** Repayment Probability $P \ge 85\%$ (Score $\ge 810$).
* **Medium Risk:** Repayment Probability $70\% \le P < 85\%$ (Score $720 - 809$).
* **High Risk:** Repayment Probability $P < 70\%$ (Score $< 720$).

### Exact Local Linear Explanations
For logistic regression, the log-odds of repayment is an exact linear combination of feature values:
$$\ln\left(\frac{P}{1 - P}\right) = \beta_0 + \sum_{i=1}^{4} \beta_i x_i$$
* **Local Contribution:** Each feature's contribution for the specific persona is computed as $\beta_i \times x_i$.
* **Transparency:** Positive terms increase repayment odds; negative terms decrease repayment odds.
* **Distinction:** The UI clearly separates local row-level contributions from global model coefficients, and distinguishes statistical associations from causal guarantees.

---

## 4. Demographic Fairness & Disparate Impact Audit

Fairness auditing evaluates whether the trained model produces equitable approval rates across protected demographic dimensions (**Gender** and **Urban/Rural Geography**).

### Approval Threshold Rule
* **Approval Criterion:** Borrowers with estimated repayment probability $P \ge 0.70$ (Score $\ge 720$) are classified as Approved.

### Four-Fifths Rule (Disparate Impact Ratio)
For each demographic dimension:
1. Identify the group with the highest approval rate as the **Reference Group**.
2. Identify the other group as the **Comparison Group**.
3. Compute the Disparate Impact Ratio:
   $$\text{Disparate Impact Ratio} = \frac{\text{Approval Rate}_{\text{comparison}}}{\text{Approval Rate}_{\text{reference}}}$$
4. An impact ratio $\ge 0.80$ satisfies the EEOC/Four-Fifths standard under synthetic test assumptions.

### Zero-Rate Protection
If the reference group's approval rate is $0.0\%$, the ratio is marked as `Undefined` rather than generating a divide-by-zero error.

### Synthetic Audit Results ($N=1,000$ Evaluation Records)
| Audit Dimension | Reference Group | Comparison Group | Disparate Impact Ratio | Four-Fifths Rule (≥ 0.80) |
| :--- | :--- | :--- | :---: | :---: |
| **Gender** | Female ($88.22\%$) | Male ($87.79\%$) | **$0.9951$** | Pass (Synthetic) |
| **Geography** | Rural ($88.59\%$) | Urban ($87.52\%$) | **$0.9879$** | Pass (Synthetic) |

### ⚠️ Critical Governance Caveat: The Proxy Bias Dilemma
Omitting demographic variables (`gender`, `geography`) from model inputs successfully eliminates **direct disparate treatment**. However, it **does not eliminate indirect proxy discrimination**. Neutral financial features (such as liquidity reserves or formal bill payments) often mirror structural historical socioeconomic disparities. Synthetic fairness checks must never be mistaken for legal non-discriminatory compliance under ECOA or fair lending statutes.

---

## 5. Judge Demonstration Script (5-Minute Walkthrough)

When demonstrating AltScore AI to evaluators or hackathon judges, follow this concise workflow:

### Step 1: Context & Disclaimer (30 Seconds)
* Open the dashboard at `http://localhost:8501`.
* Point out the **Prototype Disclaimer** banner and explain the core thesis: enabling transparent credit intelligence for thin-file borrowers using behavioral cash-flow signals.

### Step 2: The Three Synthetic Personas (1 Minute)
Use the persona dropdown in Section 2 to compare three distinct archetypes:
1. **Ramesh Kumar (Kirana Store Owner - Rural Male):**
   - *Behavior:* Steady daily grocery sales, prompt utility bill payments, healthy savings reserve ($3.8$ months).
   - *Result:* High Income Stability ($0.92$), $100\%$ Bill Consistency.
   - *Outcome:* Estimated Repayment $94.39\%$, Illustrative Score $866$, **Low Risk**.
2. **Sunita Devi (Artisan & Tailor - Urban Female):**
   - *Behavior:* Seasonal festival earnings, $100\%$ on-time utility bills, moderate savings ($2.1$ months).
   - *Result:* Moderate Income Stability ($0.74$), solid liquidity buffer.
   - *Outcome:* Estimated Repayment $74.52\%$, Illustrative Score $747$, **Medium Risk**.
3. **Aarav Sharma (Gig Economy Courier - Urban Male):**
   - *Behavior:* Volatile weekly platform earnings, multiple missed utility payments, low savings ($0.8$ months).
   - *Result:* Low Bill Consistency ($66.7\%$), low buffer.
   - *Outcome:* Estimated Repayment $59.39\%$, Illustrative Score $656$, **High Risk**.

### Step 3: Explainable Decision Decomposition (1 Minute)
* Scroll to **Section 7: Prediction Explanation**.
* Show how the logistic regression log-odds decompose into transparent feature contributions:
  - For Ramesh: Point out how Income Stability and Liquidity Buffer provide strong positive contributions to his score.
  - For Aarav: Point out how missed bill payments and low liquidity drag down his odds.
* Emphasize the governance note: these reflect model-learned statistical associations, not definitive causal claims.

### Step 4: Interactive What-If Scenario Exploration (1.5 Minutes)
* Scroll to **Section 8: Interactive What-If Simulator** with **Aarav Sharma** selected.
* *The Pitch:* "Rather than handing a borrower a black-box rejection, AltScore AI empowers financial counseling."
* Move the **Financial Liquidity Buffer** slider from $0.8$ months up to $3.5$ months.
* Move the **Bill-Payment Consistency** slider from $0.67$ up to $1.00$.
* Observe the real-time recalculated card:
  - Probability rises from $59.4\%$ to $\approx 78\%$.
  - Score climbs from $656$ to $\approx 770$.
  - Risk category moves from **High Risk** to **Medium Risk**.
  - Show the live contribution shift in the adjusted explanation chart.

### Step 5: Demographic Fairness Audit & Caveats (1 Minute)
* Scroll to **Section 9: Synthetic Fairness & Disparate Impact Audit**.
* Review the Disparate Impact Ratios:
  - Gender: $0.9951$ (Passes Four-Fifths rule $\ge 0.80$).
  - Geography: $0.9879$ (Passes Four-Fifths rule $\ge 0.80$).
* Deliver the concluding takeaway on **Proxy Bias**: explain why blind omission of demographic columns is necessary but insufficient, and why ongoing demographic monitoring is mandatory in responsible lending.

---

## 6. Installation & Execution Guide

### Prerequisites
* Windows, macOS, or Linux
* Python 3.10, 3.11, 3.12, 3.13, or 3.14

### Setup & Run Steps (PowerShell / Terminal)

```powershell
# 1. Clone the repository
git clone https://github.com/commit-gaurav/credit-score.git
cd credit-score

# 2. Set up virtual environment (if not already created)
python -m venv .venv

# 3. Activate the environment
# Windows PowerShell:
.\.venv\bin\Activate.ps1
# Windows Command Prompt:
# .\.venv\Scripts\activate.bat
# Linux / macOS:
# source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the Streamlit application
streamlit run app.py
```

The application will launch at `http://localhost:8501`.

---

## 7. Development Slice Checklist

- [x] **Slice 1:** App shell, architecture, and prominent prototype disclaimer (`6c620e3`)
- [x] **Slice 2:** Synthetic personas (Ramesh, Sunita, Aarav) & transaction ledgers (`fc5b7fb`)
- [x] **Slice 3:** Behavioral feature extraction engine with unit definitions & caps (`9cf4851`)
- [x] **Slice 4:** Reproducible synthetic dataset generation ($N=1,000$, seed=42) (`6315a5c`)
- [x] **Slice 5:** Baseline logistic regression training & held-out test evaluation (`fda2362`)
- [x] **Slice 6:** Persona probability prediction, 300–900 score mapping & risk tiers (`0a2299e`)
- [x] **Slice 7:** Exact local linear prediction explanation ($\beta_0 + \sum \beta_i x_i$) (`cb2009a`)
- [x] **Slice 8:** Interactive 4-feature what-if scenario simulator & comparison (`8b595c7`)
- [x] **Slice 9:** Synthetic demographic fairness audit & disparate impact analysis (`6fcf2b5`)
- [x] **Slice 10:** Comprehensive documentation polish, demo script, and complete run-through

---

## 8. License & Attribution

This project is released under the **MIT License** for open research and educational exploration. Designed for hackathons, academic research in AI explainability, and algorithmic fairness audits.
