# AltScore AI — Hackathon Plan

## 1. PROJECT OVERVIEW

AltScore AI uses synthetic transaction records and explainable machine learning to estimate repayment risk and explore financial scenarios.
It is for hackathon judges evaluating a learning prototype for gig workers, small shopkeepers, rural entrepreneurs, and first-time borrowers.
It explores alternative financial signals for people with limited credit histories while showing prediction explanations and potential group disparities.

## 2. TECH STACK

- **Python** — one approachable language for the dashboard, feature calculations, and model.
- **Streamlit** — builds a local interactive dashboard without a separate frontend or backend.
- **pandas** — organizes synthetic transactions and calculated feature values in tables.
- **scikit-learn** — trains, splits, and evaluates a simple, explainable classifier.
- **NumPy** — generates reproducible synthetic training and evaluation records.
- **Plotly (optional)** — displays clear interactive charts if time allows; use Streamlit tables/metrics first.
- **Git** — records working checkpoints and supports recovery from mistakes.
- **No database or external integrations** — all data is synthetic and held in memory, keeping the demo local and easy to run.

## 3. FEATURES

### Must-have

- Select among synthetic profiles for Ramesh Kumar, Sunita Devi, and Aarav Sharma.
- Display each persona's synthetic transaction history and extract four consistently defined features:
  1. **Income stability:** normalized measure of how consistent monthly income is over the available months (0–1; higher means more stable).
  2. **Bill-payment consistency:** fraction of recorded bills paid on time (0–1).
  3. **Payment regularity:** fraction of expected income/payment periods with a recorded incoming payment (0–1).
  4. **Financial liquidity buffer:** liquid balance divided by average monthly essential outflow, capped at 6 months (0–6 months).
- Document the feature formulas and simple missing-data handling; missing transactions must not silently be treated as missed payments.
- Generate a reproducible synthetic training set with a documented simulated repayment target, excluding demographic and target columns from model inputs.
- Train and evaluate a simple classifier using a held-out split; report appropriate metrics as synthetic-data results.
- Display repayment probability, a 300–900 illustrative score calculated as 300 + 600 × repayment probability, risk category, and a logistic-regression explanation. The score is bounded by the probability range and is not an established credit score.
- Provide interactive what-if controls that update the prediction, score, risk category, and explanation using the actual trained model.
- Calculate approval rates and disparate impact ratios for both gender and urban/rural groups. For each comparison, use the higher-rate group as reference and lower-rate group as comparison; ratio = comparison rate divided by reference rate. If reference rate is zero, show undefined.
- Clearly state that predictions, scores, and fairness results are synthetic demonstrations, not real credit decisions or evidence of real-world performance/fairness.

### Nice-to-have

- SHAP explanation after a correct baseline explanation works and can be explained.
- Raw SMS parsing, only as an offline mock with wholly synthetic messages.
- Synthetic transaction generator, extra personas, or advanced visualizations.
- Fairness mitigation, with limitations explained.
- Fraud detection or Account Aggregator integration only if all core goals are already met (unlikely within this time budget).

## 4. FOLDER STRUCTURE

~~~text
altscore-ai/
├── PLAN.md                    — Editable build plan, slice checklist, and decision log.
├── README.md                  — Installation, run instructions, demo script, and limitations.
├── requirements.txt           — Minimal Python dependencies for the app.
├── app.py                     — Streamlit dashboard and user interaction flow.
├── data/
│   ├── __init__.py             — Makes the data folder importable as a Python package.
│   ├── personas.py             — Three synthetic personas and their transaction records.
│   └── synthetic_data.py       — Reproducible training and fairness-evaluation data generation.
├── features/
│   ├── __init__.py             — Makes the feature folder importable as a Python package.
│   └── extraction.py           — Definitions and calculations for the four financial features.
├── model/
│   ├── __init__.py             — Makes the model folder importable as a Python package.
│   └── scoring.py              — Model training, evaluation, prediction, score mapping, and explanation.
├── fairness/
│   ├── __init__.py             — Makes the fairness folder importable as a Python package.
│   └── audit.py                — Group approval rates and disparate impact calculation.
└── .gitignore                  — Excludes virtual environments, caches, and local generated artifacts.
~~~

This is the planned end-state tree, not a requirement to create every file at once. Data and model objects stay in memory; no database or generated data file is needed for the core demo.

## 5. DATABASE DESIGN

**N/A.** There is no database. Synthetic transactions, generated training/evaluation data, and model objects are created reproducibly in memory at app startup. No real user information is collected or persisted.

## 6. PAGES / ROUTES / API ENDPOINTS

The application has one local Streamlit page and no separate HTTP API endpoints.

| Name | Purpose | Inputs | Outputs |
|---|---|---|---|
| AltScore AI dashboard (app.py, root page) | Select a persona; inspect transactions and extracted features; view prediction and explanation; try what-if changes; inspect fairness results. | Persona selection; optional what-if values for the four financial features. Fairness group attributes are generated internally and are not model inputs. | Synthetic transaction summary; four feature values; repayment probability; illustrative score and risk category; model-linked explanation; updated what-if results; group approval rates and disparate impact ratio; limitations disclaimer. |

## 7. KEY FUNCTIONS / MODULES

| Name | What it does | File |
|---|---|---|
| get_personas() | Returns the three named profiles and their wholly synthetic transaction records and demo group attributes. | data/personas.py |
| generate_synthetic_dataset(seed, row_count) | Creates reproducible feature rows, demographic evaluation attributes, and documented simulated target labels. | data/synthetic_data.py |
| extract_features(transactions) | Calculates income stability, bill-payment consistency, payment regularity, and liquidity buffer using documented formulas and missing-data rules. | features/extraction.py |
| train_model(data, seed) | Separates features and target, makes a held-out split, fits a simple classifier, and returns the fitted model and evaluation results. | model/scoring.py |
| predict_repayment_probability(model, features) | Returns the model's estimated probability for the simulated positive repayment outcome. | model/scoring.py |
| probability_to_score(probability) | Applies the documented illustrative linear mapping from probability to a bounded 300–900 score. | model/scoring.py |
| categorize_risk(probability) | Maps probability ranges to documented demo risk categories. | model/scoring.py |
| explain_prediction(model, features) | Provides an explanation tied to the selected model (local contributions for a linear model, or a clearly identified model-based explanation for a small tree). | model/scoring.py |
| calculate_fairness_metrics(model, evaluation_data, threshold) | Calculates group approval rates and the defined disparate impact ratio; handles zero reference rates explicitly. | fairness/audit.py |
| main() | Builds the dashboard, obtains selections and control values, calls modules, and renders outputs. | app.py |

## 8. DATA FLOW

1. The user opens the local Streamlit page; app.py starts main() and creates the reproducible in-memory dataset and model.
2. The app calls get_personas() and displays the three synthetic profiles. The user selects one.
3. The app loads that persona's synthetic transaction records and passes them to extract_features().
4. extract_features() computes the four defined values with consistent units and missing-data handling; the app displays both the values and a brief definition.
5. Training data is generated with a fixed seed. Its synthetic target-generation process is documented. Demographic fields and target labels are kept separate from model input columns.
6. train_model() makes a held-out split and fits the selected simple classifier on training rows. It returns the model and evaluation metrics; the app labels these as results on synthetic data.
7. The app sends the selected persona's four features to predict_repayment_probability(). It converts the probability to an illustrative score and risk category using documented functions, then obtains a model-linked explanation.
8. The user changes one or more what-if controls. The app sends only the changed financial features through the same prediction, score, category, and explanation functions, then displays the baseline and changed results together.
9. For the audit, the app sends synthetic evaluation rows to calculate_fairness_metrics(). The audit uses demographic labels only to group predictions after model scoring, calculates actual approval rates and the specified disparate impact ratio, and handles zero rates by displaying the ratio as undefined rather than inventing a value.
10. The page displays the audit alongside the reference/comparison group definitions and limitations: synthetic comparisons do not establish real-world fairness, and omitting demographics from model inputs does not remove proxy bias.
11. No step writes to a database or stores real personal data.

## 9. CODING RULES

- Use simple, readable Python and avoid fancy patterns or unnecessary abstractions.
- Add comments for non-obvious calculations, synthetic label generation, score mapping, and fairness metrics.
- Use snake_case for variables and functions, PascalCase for classes only if classes are necessary, and UPPER_SNAKE_CASE for constants.
- Give each function one responsibility; keep functions small and name them for the result they return.
- Keep dashboard code in app.py and data, feature, model, and fairness responsibilities in their planned modules.
- Use fixed random seeds for reproducible generated data and train/test splitting.
- Keep demographic attributes and target labels out of the model's prediction features.
- State feature units, formulas, score mapping, risk thresholds, approval threshold, reference/comparison groups, and missing-data rules in code comments or README.
- Every displayed number must come from data or a calculation; never assign a predetermined score or prediction to a persona.
- Label all outputs as synthetic demonstrations; distinguish model associations from causes and synthetic test metrics from real-world performance.
- Add no new dependency or service without a clear core-feature need.

## 10. BUILD ORDER

Each slice is small and ends with an app that can be launched. Continue from the last working result; each next slice should preserve prior behavior.

1. **App shell and disclaimer.** Create requirements.txt, app.py, and a minimal README. Show the project purpose and prototype disclaimer. **Runnable result:** Streamlit launches the shell locally.
2. **Synthetic personas and transactions.** Add data/personas.py; show persona selection and a compact transaction table. **Runnable result:** switching profiles changes the shown synthetic records.
3. **Feature definitions and extraction.** Add features/extraction.py; implement and display all four features with units and definitions. **Runnable result:** selecting a persona shows computed values.
4. **Reproducible training data.** Add data/synthetic_data.py with documented target labels, fixed seed, and separated demographic fields. **Runnable result:** app shows a small generated-data summary.
5. **Train and evaluate a baseline.** Add model/scoring.py; use a held-out split and a simple classifier, showing appropriate synthetic test metrics. **Runnable result:** app launches and displays actual computed metrics.
6. **Persona prediction and score.** Display estimated probability, documented illustrative score mapping, and risk category for the selected profile. **Runnable result:** changing persona recalculates model outputs.
7. **Prediction explanation.** Add an explanation suitable for the chosen model and distinguish local explanation from general feature importance. **Runnable result:** each prediction has an explanation derived from model behavior.
8. **What-if simulator.** Add controls for selected financial features and run changed values through the actual prediction pipeline. **Runnable result:** changing controls updates probability, score, category, and explanation beside the baseline.
9. **Fairness audit.** Add gender and urban/rural approval rates, defined disparate impact ratios, zero-rate handling, and proxy-bias caveat. **Runnable result:** audit metrics are computed from synthetic evaluation rows and displayed with group definitions.
10. **Finish demo and documentation.** Improve only essential layout, finish setup instructions, limitations, and a short demo script in README. Run through the complete workflow and prepare judge explanations. **Runnable result:** a fresh local launch supports the end-to-end demonstration.

## 11. PROGRESS LOG

- [x] Slice 1 — App shell and disclaimer — Completed (`6c620e3`)
- [x] Slice 2 — Synthetic personas and transactions — Completed (`fc5b7fb`)
- [x] Slice 3 — Feature definitions and extraction — Completed (`9cf4851`)
- [x] Slice 4 — Reproducible training data — Completed (`6315a5c`)
- [x] Slice 5 — Model training and held-out evaluation — Completed (`fda2362`)
- [x] Slice 6 — Persona prediction, score, and category — Completed (`0a2299e`)
- [x] Slice 7 — Prediction explanation — Completed (`cb2009a`)
- [x] Slice 8 — What-if simulator — Completed (`8b595c7`)
- [x] Slice 9 — Fairness audit — Completed (`6fcf2b5`)
- [x] Slice 10 — Demo, documentation, and final run-through — Completed

### CURRENT STATE

- **Last completed slice:** Slice 10 (Full Prototype Active).
- **What is half-done:** Nothing; all 10 slices implemented and verified end-to-end.
- **Known bugs:** None; all imports, metrics, what-if recalculations, explanations, and fairness ratios verify cleanly.
- **Next step:** Present full demonstration walkthrough to hackathon judges.

## 12. DECISIONS LOG

- **Single Python and Streamlit application:** fits the six-hour solo implementation window and is straightforward to explain.
- **No database or external integrations:** keeps the demo local, reproducible, and free of real financial data.
- **Four behavior-based financial features:** follows the brief's feature definitions and excludes demographic attributes from model inputs.
- **Synthetic labels and evaluation only:** every training and audit result is demonstrative; synthetic performance cannot support real-world claims.
- **Use logistic regression:** it provides repayment probabilities for the score and feature contributions that can be explained to judges. This is a choice for a credible, teachable demo, not a guarantee of higher accuracy or winning.
- **Keep prediction explanation distinct from global importance:** explain the selected model's behavior for the current row where feasible, and label general feature importance correctly.
- **Define scores and fairness metrics explicitly:** the 300–900 score is an illustrative transformation; fairness group rates and disparate impact must use stated thresholds and actual computed values.
- **Build core workflow before optional work:** defer SHAP, SMS parsing, mitigation, extra personas, and integrations until the working prediction, simulator, and audit are complete.



