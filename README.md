# ⚠️ Construction Delay Risk Predictor

> Predicts the probability of construction project delays using machine learning — enabling proactive risk management before delays occur.

A research-grade Python application that applies **Gradient Boosting classification** to predict delay risk from project characteristics known at initiation, combined with **Monte Carlo simulation** for probabilistic duration forecasting.

Built as an MSc application portfolio project in Construction Management \& Engineering.

\---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Methodology](#methodology)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Example Input \& Output](#example-input--output)
8. [Dashboard Screenshots](#dashboard-screenshots)
9. [Research Contributions](#research-contributions)
10. [Future Work](#future-work)
11. [License](#license)

\---

## Overview

Construction delays cost the global industry an estimated **$1.6 trillion annually**. Over 65% of projects worldwide finish late. This system provides a **pre-project risk assessment tool** that:

* Predicts delay probability from 15 project features
* Identifies the top contributing risk factors with impact scores
* Simulates probabilistic completion dates via Monte Carlo
* Benchmarks against global industry delay rates
* Enables what-if mitigation scenario analysis

**Key Model Performance (Gradient Boosting, 1,200 projects):**

|Metric|Value|
|-|-|
|ROC-AUC|0.87|
|F1 Score|0.83|
|Precision|0.84|
|Recall|0.82|
|CV AUC (5-fold)|0.86 ± 0.02|

\---

## Problem Statement

Construction delay prediction is a **binary classification problem**: given features known at project initiation, predict whether the project will exceed its planned duration by more than 10%.

Key challenges:

1. **Multicollinearity** — risk factors (weather, labour, materials) are correlated
2. **Class imbalance** — delay rates vary significantly by region and project type
3. **Interpretability** — project managers need to understand *why* a project is high risk, not just the probability

\---

## Methodology

### Machine Learning Models

Three classifiers are trained and benchmarked:

|Model|Role|
|-|-|
|Gradient Boosting|Primary (best AUC)|
|Random Forest|Benchmark ensemble|
|Logistic Regression|Interpretable baseline|

All models use a **scikit-learn Pipeline** with:

* `StandardScaler` for numeric features
* `OneHotEncoder` for categorical features
* 5-fold stratified cross-validation

### Feature Set (15 Variables)

|Feature|Type|Description|
|-|-|-|
|project\_size\_m2|Numeric|Building footprint|
|num\_workers|Numeric|Peak workforce|
|num\_subcontractors|Numeric|Specialist firms|
|planned\_duration\_days|Numeric|Scheduled duration|
|budget\_usd|Numeric|Approved budget|
|schedule\_buffer\_days|Numeric|Contingency days|
|weather\_risk\_score|Numeric (1–10)|Climate/season risk|
|material\_delivery\_risk|Numeric (1–10)|Procurement risk|
|labour\_availability|Numeric (1–10)|Local workforce availability|
|design\_complexity|Numeric (1–10)|Architectural complexity|
|site\_accessibility|Numeric (1–10)|Site access conditions|
|previous\_delays|Numeric (0–4)|Historical delay count|
|contract\_type|Categorical|Fixed Price / Cost Plus / Design-Build|
|project\_type|Categorical|Residential / Commercial / Infrastructure / Industrial|
|region|Categorical|North / South / East / West / Central|

### Delay Probability Model

The synthetic dataset is generated using a logistic ground-truth model:

```
log-odds = −3.5
         + 0.25 × weather\_risk
         + 0.30 × material\_delivery\_risk
         − 0.20 × labour\_availability
         + 0.22 × design\_complexity
         + 0.40 × previous\_delays
         − 0.04 × schedule\_buffer\_days
         + 0.20 × \[Fixed Price contract]
         + ε
```

### Monte Carlo Simulation

For each project, 10,000 completion scenarios are sampled:

* **Delayed projects**: lognormal overrun factor (μ=1.25, σ=0.20)
* **On-time projects**: normal variation (μ=1.0, σ=0.05)

Returns P50, P80, P90 completion estimates and on-time probability.

\---

## Project Structure

```
construction-delay-predictor/
│
├── data/
│   └── construction\_projects.csv     ← 1,200 synthetic project records
│
├── models/
│   ├── delay\_model.pkl                ← Trained Gradient Boosting pipeline
│   ├── model\_evaluation.json          ← Performance metrics for all models
│   └── feature\_importance.csv         ← Feature importance rankings
│
├── notebooks/
│   └── delay\_prediction\_demo.ipynb    ← Full analysis notebook
│
├── src/
│   ├── data\_generator.py              ← Synthetic dataset generation
│   ├── train\_model.py                 ← ML training, evaluation, prediction API
│   └── risk\_analysis.py               ← Monte Carlo, portfolio analysis, scenarios
│
├── app/
│   └── streamlit\_app.py               ← Interactive risk assessment dashboard
│
├── assets/                            ← Dashboard screenshots
├── requirements.txt
├── README.md
└── LICENSE
```

\---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ghost0d0/construction-delay-predictor.git
cd construction-delay-predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt
```

\---

## Usage

### Stage 1 — Generate Dataset

```bash
python src/data\_generator.py
```

Output: `data/construction\_projects.csv` (1,200 projects)

### Stage 2 — Train Model

```bash
python src/train\_model.py
```

Output: `models/delay\_model.pkl`, `models/model\_evaluation.json`, `models/feature\_importance.csv`

### Stage 3 — Launch Dashboard

```bash
streamlit run app/streamlit\_app.py
```

Open: `http://localhost:8501`

### Optional — Notebook

```bash
jupyter notebook notebooks/delay\_prediction\_demo.ipynb
```

\---

## Example Input \& Output

### Input Project Profile

```
Project Type      : Commercial
Contract Type     : Fixed Price
Planned Duration  : 180 days
Budget            : $12,000,000
Workers           : 45
Subcontractors    : 8
Weather Risk      : 7 / 10
Material Risk     : 8 / 10
Labour Avail.     : 4 / 10
Schedule Buffer   : 10 days
Previous Delays   : 2
```

### Output

```
Delay Risk:  68%  \[HIGH]

Top Risk Factors:
  • Material procurement delay     (impact: 80%)
  • Low labour availability        (impact: 60%)
  • Adverse weather conditions     (impact: 70%)
  • History of previous delays     (impact: 50%)
  • Fixed price contract pressure  (impact: 55%)

Monte Carlo Forecast:
  P50 completion : 198 days
  P80 completion : 231 days
  P90 completion : 252 days
  On-time prob   : 27%
```

\---

## Dashboard Screenshots

### Risk Assessment

!\[Risk](assets/risk.png)

### Model Performance

!\[Model](assets/model.png)

### Portfolio Analysis

!\[Portfolio](assets/portfolio.png)

### Monte Carlo Simulation

!\[Simulation](assets/simulation.png)

### Scenario Analysis

!\[Scenarios](assets/scenarios.png)

\---

## Research Contributions

1. **Pre-project Risk Scoring** — Delay prediction at project initiation, before any construction begins, enabling early risk mitigation.
2. **Ensemble ML Benchmarking** — Systematic comparison of Gradient Boosting, Random Forest, and Logistic Regression with cross-validation.
3. **Explainable Risk Factors** — Rule-based factor identification converts black-box probabilities into actionable management insights.
4. **Probabilistic Forecasting** — Monte Carlo simulation translates delay probability into P50/P80/P90 duration estimates aligned with construction industry risk standards.
5. **Industry Benchmarking** — Portfolio analysis benchmarked against global delay statistics (65% global delay rate, 28% average cost overrun).

\---

## Future Work

* **XGBoost / LightGBM**: Benchmark against gradient boosting variants for performance gains
* **SHAP Values**: Replace rule-based explanations with model-faithful SHAP attributions
* **Real Project Data**: Validate on publicly available datasets (e.g., Dodge Data, Rider Levett Bucknall)
* **Time-Series Risk**: Dynamic risk updating as project progresses using survival analysis
* **NLP Risk Signals**: Extract early warning signals from project meeting minutes using NLP
* **Cost Overrun Prediction**: Extend to regression model predicting % cost overrun
* **BIM Integration**: Automatic feature extraction from Building Information Models

\---

## Technologies

|Category|Library|
|-|-|
|ML Models|scikit-learn (GradientBoosting, RandomForest, LogisticRegression)|
|Data|pandas, numpy|
|Simulation|numpy (Monte Carlo)|
|Visualisation|matplotlib, seaborn|
|Dashboard|Streamlit|
|Notebook|Jupyter|

\---

## License

MIT License — see [LICENSE](LICENSE) for details.

\---

## Author

Developed as an MSc research portfolio project in Construction Management \& Engineering.
Demonstrates machine learning applications in construction project risk management.

\---

*Construction Delay Risk Predictor · ML Risk Modelling · Construction Management · 2024*

