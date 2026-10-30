# AltScore AI

An explainable alternative credit scoring and scenario exploration prototype built for hackathon demonstration.

## Project Purpose

Traditional credit bureau scores often overlook individuals without formal banking or credit histories, such as gig workers, small shopkeepers, and first-time borrowers. **AltScore AI** explores how alternative financial behavioral signals (such as income stability, bill payment consistency, payment regularity, and liquidity buffers) can be modeled using explainable machine learning to provide transparent risk estimations, interactive what-if simulations, and fair lending audits.

> **IMPORTANT DISCLAIMER:**
> AltScore AI is a synthetic research prototype created strictly for educational and evaluation purposes. All data, scores, risk classifications, and fairness metrics are demonstrative simulations. It is **not** a real credit decisioning tool, does not access real consumer data, and must not be used for actual lending decisions.

---

## Tech Stack (Current Slice)

- **Language:** Python 3.10+
- **Frontend / Dashboard:** Streamlit
- *(Planned upcoming additions: pandas, scikit-learn, NumPy)*

---

## Installation & Running

### 1. Activate the Virtual Environment

The virtual environment `.venv` has already been created with all pre-compiled dependencies. In PowerShell, activate it:

```powershell
.\.venv\bin\Activate.ps1
```

*(Note: Depending on your PowerShell execution policy, you can also run directly without activating via `.\.venv\bin\streamlit run app.py` or `.\.venv\bin\python -m streamlit run app.py`)*

### 2. Run the Streamlit App

```powershell
.\.venv\bin\python -m streamlit run app.py
```

The app will launch in your default web browser (typically at `http://localhost:8501`).

---

## Current Status (Slice 1)

- [x] **Slice 1: App Shell & Disclaimer** — Minimal runnable Streamlit application shell displaying project identity, purpose statement, and prominent synthetic prototype disclaimer.
- [ ] Slice 2: Synthetic personas and transactions
- [ ] Slice 3: Feature definitions and extraction
- [ ] Slice 4: Reproducible training data
- [ ] Slice 5: Model training and held-out evaluation
- [ ] Slice 6: Persona prediction and score
- [ ] Slice 7: Prediction explanation
- [ ] Slice 8: What-if simulator
- [ ] Slice 9: Fairness audit
- [ ] Slice 10: Demo and documentation polish

