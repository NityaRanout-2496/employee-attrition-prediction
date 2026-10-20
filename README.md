# Employee Attrition Prediction and Risk Scoring System

Machine learning system that predicts which employees are likely to leave Palo Alto Networks, scores every employee as Low, Medium or High risk, explains the main drivers, and presents everything in an interactive Streamlit dashboard.

**Live dashboard:** https://palo-alto-attrition-risk.streamlit.app

## Problem

HR teams can describe past attrition but cannot tell which employees are likely to leave next. This leads to sudden resignations, loss of high performers and counter-offers that come too late. This project predicts attrition risk early, so retention can be proactive.

## Dataset

- 1,470 employees, 31 columns (`data/Palo_Alto_Networks.csv`)
- Target: `Attrition` (237 left, 16.1%; 1,233 stayed)
- No missing values, no duplicate rows

## Approach

1. **EDA:** attrition by overtime, job role, income, travel, marital status and correlations
2. **Preprocessing:** encoding, stratified 80/20 split, scaling (fit on train only), SMOTE (train only)
3. **Feature engineering:** IncomeToExperience, PromotionDelay, EngagementScore, WorkloadStress
4. **Models:** Logistic Regression, Random Forest, XGBoost, compared with a test split and 5-fold cross-validation
5. **Threshold tuning:** decision threshold 0.35, chosen on training data only
6. **Risk scoring:** cross-validated probabilities for all employees, grouped as Low (<30%), Medium (30-60%), High (>60%)
7. **Explainability:** grouped driver importance and plain-English reason codes per employee
8. **Dashboard:** five-module Streamlit app

## Key results

| Model | Test ROC-AUC | CV ROC-AUC | CV Recall | CV Precision |
|---|---|---|---|---|
| Logistic Regression | 0.813 | 0.833 | 0.495 | 0.758 |
| XGBoost | 0.750 | 0.794 | 0.395 | 0.599 |
| Random Forest | 0.739 | 0.769 | 0.332 | 0.632 |

Logistic Regression was selected. At the 0.35 threshold it caught 24 of the 47 leavers in the test set (precision 0.51, recall 0.51).

| Risk category | Employees | Actual attrition rate |
|---|---|---|
| Low | 1,158 | 8.1% |
| Medium | 209 | 30.1% |
| High | 103 | 77.7% |

**Main drivers of risk:** low satisfaction and engagement, overtime, long time since last promotion, and high-turnover job roles (for example Sales Representatives, 39.8% attrition).

## Dashboard modules

- Attrition risk dashboard (risk distribution, high-risk counts)
- Employee risk profile (probability, gauge, top reasons)
- Department and role risk view
- Explainability panel (grouped driver importance)
- What-if scenario (change overtime, income, satisfaction and more, then re-score)

Sidebar filters: department, job role and minimum attrition probability.

## Project structure

```
attrition-project/
├── app.py                  Streamlit dashboard
├── requirements.txt        Libraries for the dashboard
├── data/
│   ├── Palo_Alto_Networks.csv
│   └── processed/          Scores, features, importance table
├── models/                 Saved model, scaler, threshold, column list
└── notebooks/
    ├── 01_eda.ipynb
    ├── 02_preprocessing.ipynb
    ├── 03_modeling.ipynb
    └── 04_risk_scoring.ipynb
```

## Run it locally

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To rerun the notebooks, also install: `pip install jupyter ipykernel matplotlib seaborn imbalanced-learn xgboost`

## Limitations

- Small dataset: only 47 leavers in the test set, so metrics are noisy
- The data is a snapshot, so findings are associations, not proof of cause
- Risk scores support HR judgement and should not be the only basis for any decision
- Gender and marital status are model inputs and should be reviewed for fairness before wider use
