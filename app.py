import os

import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Attrition Risk Dashboard", layout="wide")


@st.cache_data
def load_scores():
    return pd.read_csv("data/processed/risk_scores.csv")


scores = load_scores()

st.title("Employee Attrition Risk Dashboard")
st.caption("Palo Alto Networks - ML-based attrition prediction and risk scoring")

# ---- Sidebar filters ----
st.sidebar.header("Filters")

departments = sorted(scores["Department"].unique())
sel_dept = st.sidebar.multiselect("Department", departments, default=departments)

roles_available = sorted(scores[scores["Department"].isin(sel_dept)]["JobRole"].unique())
sel_role = st.sidebar.multiselect("Job role", roles_available, default=roles_available)

min_prob = st.sidebar.slider("Minimum attrition probability", 0.0, 1.0, 0.0, 0.05)

filtered = scores[
    scores["Department"].isin(sel_dept)
    & scores["JobRole"].isin(sel_role)
    & (scores["AttritionProbability"] >= min_prob)
]

if filtered.empty:
    st.warning("No employees match these filters. Try widening them.")
    st.stop()

# ---- Key numbers ----
total = len(filtered)
high = (filtered["RiskCategory"] == "High").sum()
medium = (filtered["RiskCategory"] == "Medium").sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees shown", f"{total:,}")
c2.metric("High-risk employees", f"{high:,}")
c3.metric("Medium-risk employees", f"{medium:,}")
c4.metric("Average attrition probability", f"{filtered['AttritionProbability'].mean():.1%}")

# ---- Risk distribution chart ----
st.subheader("Overall risk distribution")

order = ["Low", "Medium", "High"]
counts = (
    filtered["RiskCategory"]
    .value_counts()
    .reindex(order, fill_value=0)
    .reset_index()
)
counts.columns = ["Risk category", "Employees"]

fig = px.bar(
    counts,
    x="Risk category",
    y="Employees",
    color="Risk category",
    color_discrete_map={"Low": "seagreen", "Medium": "orange", "High": "indianred"},
    text="Employees",
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ---- Employee risk profile ----
st.divider()
st.subheader("Employee risk profile")

emp_id = st.selectbox("Select Employee ID", filtered["EmployeeID"].tolist())
emp = filtered[filtered["EmployeeID"] == emp_id].iloc[0]

p1, p2, p3 = st.columns(3)
p1.metric("Attrition probability", f"{emp['AttritionProbability']:.1%}")
p2.metric("Risk category", emp["RiskCategory"])
p3.metric("Job role", emp["JobRole"])

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=emp["AttritionProbability"] * 100,
    number={"suffix": "%"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "darkgray"},
        "steps": [
            {"range": [0, 30], "color": "seagreen"},
            {"range": [30, 60], "color": "orange"},
            {"range": [60, 100], "color": "indianred"},
        ],
    },
))
gauge.update_layout(height=280, margin=dict(l=30, r=30, t=30, b=10))
st.plotly_chart(gauge, use_container_width=True)

st.markdown("**Main factors raising this employee's risk:**")
for reason in str(emp["TopReasons"]).split(" + "):
    st.write("- " + reason)

# ---- Department and role risk view ----
st.divider()
st.subheader("Department and role risk view")

dept = (
    filtered.groupby("Department")
    .agg(
        Employees=("EmployeeID", "count"),
        AvgRisk=("AttritionProbability", "mean"),
        HighRisk=("RiskCategory", lambda s: (s == "High").sum()),
    )
    .reset_index()
)
dept["Average risk (%)"] = (dept["AvgRisk"] * 100).round(1)

role = (
    filtered.groupby("JobRole")
    .agg(
        Employees=("EmployeeID", "count"),
        AvgRisk=("AttritionProbability", "mean"),
        HighRisk=("RiskCategory", lambda s: (s == "High").sum()),
    )
    .reset_index()
    .sort_values("AvgRisk")
)
role["Average risk (%)"] = (role["AvgRisk"] * 100).round(1)

left, right = st.columns(2)

with left:
    st.markdown("**Average attrition risk by department**")
    fig_d = px.bar(
        dept, x="Department", y="Average risk (%)",
        text="Average risk (%)", color_discrete_sequence=["steelblue"],
    )
    st.plotly_chart(fig_d, use_container_width=True)

with right:
    st.markdown("**Average attrition risk by job role**")
    fig_r = px.bar(
        role, x="Average risk (%)", y="JobRole", orientation="h",
        text="Average risk (%)", color_discrete_sequence=["steelblue"],
    )
    st.plotly_chart(fig_r, use_container_width=True)

st.markdown("**Details by job role**")
st.dataframe(
    role[["JobRole", "Employees", "HighRisk", "Average risk (%)"]]
    .sort_values("Average risk (%)", ascending=False),
    hide_index=True,
    use_container_width=True,
)

# ---- Explainability panel ----
st.divider()
st.subheader("Explainability: what drives attrition risk")

imp_path = "data/processed/feature_importance.csv"
if os.path.exists(imp_path):
    imp = pd.read_csv(imp_path)
    fig_i = px.bar(
        imp.sort_values("Importance"),
        x="Importance",
        y="Driver",
        orientation="h",
        color_discrete_sequence=["steelblue"],
    )
    fig_i.update_layout(height=450)
    st.plotly_chart(fig_i, use_container_width=True)
    st.caption(
        "Bigger bar = this factor moves employees' risk scores more, on average. "
        "Related columns (for example the four satisfaction ratings) are combined into one driver."
    )
else:
    st.info("feature_importance.csv not found. Run Part A of App Step 6 in the 04_risk_scoring notebook first.")

# ---- What-if scenario ----
st.divider()
st.subheader("What-if scenario")

feat_path = "data/processed/employee_features.csv"
if not os.path.exists(feat_path):
    st.info("employee_features.csv not found. Run Part A of App Step 7 in the 04_risk_scoring notebook first.")
    st.stop()

NUM_COLS = [
    "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears",
    "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
    "IncomeToExperience", "PromotionDelay", "EngagementScore",
]


@st.cache_resource
def load_model():
    model = joblib.load("models/logistic_regression.pkl")
    scaler = joblib.load("models/scaler.pkl")
    cols = joblib.load("models/feature_columns.pkl")
    return model, scaler, cols


@st.cache_data
def load_features():
    return pd.read_csv("data/processed/employee_features.csv")


model, scaler, feature_cols = load_model()
feats = load_features()


def predict_prob(row_df):
    x = row_df[feature_cols].copy()
    x[NUM_COLS] = scaler.transform(x[NUM_COLS])
    return float(model.predict_proba(x)[:, 1][0])


def category(p):
    if p < 0.30:
        return "Low"
    if p <= 0.60:
        return "Medium"
    return "High"


st.caption(
    f"Showing Employee {emp_id} (selected in the profile above). "
    "Change the settings below and the model re-scores this employee."
)

base = feats[feats["EmployeeID"] == emp_id].iloc[[0]].copy()

w1, w2, w3 = st.columns(3)
with w1:
    ot = st.checkbox("Works overtime", value=bool(base["OverTime"].iloc[0]), key=f"ot_{emp_id}")
    income = st.number_input(
        "Monthly income", min_value=1000, max_value=20000,
        value=int(base["MonthlyIncome"].iloc[0]), step=500, key=f"inc_{emp_id}",
    )
with w2:
    js = st.slider("Job satisfaction (1-4)", 1, 4, int(base["JobSatisfaction"].iloc[0]), key=f"js_{emp_id}")
    es = st.slider("Environment satisfaction (1-4)", 1, 4, int(base["EnvironmentSatisfaction"].iloc[0]), key=f"es_{emp_id}")
with w3:
    wlb = st.slider("Work-life balance (1-4)", 1, 4, int(base["WorkLifeBalance"].iloc[0]), key=f"wlb_{emp_id}")
    ysp = st.slider("Years since last promotion", 0, 15, int(base["YearsSinceLastPromotion"].iloc[0]), key=f"ysp_{emp_id}")

new = base.copy()
new["OverTime"] = int(ot)
new["MonthlyIncome"] = income
new["JobSatisfaction"] = js
new["EnvironmentSatisfaction"] = es
new["WorkLifeBalance"] = wlb
new["YearsSinceLastPromotion"] = ysp

# recompute the engineered features from the changed values
new["EngagementScore"] = new[
    ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "JobInvolvement"]
].mean(axis=1)
new["PromotionDelay"] = new["YearsSinceLastPromotion"] / (new["YearsAtCompany"] + 1)
new["IncomeToExperience"] = new["MonthlyIncome"] / (new["TotalWorkingYears"] + 1)
new["WorkloadStress"] = (
    (new["OverTime"] == 1) & (new["BusinessTravel_Travel_Frequently"] == 1)
).astype(int)

base_p = predict_prob(base)
new_p = predict_prob(new)

r1, r2, r3 = st.columns(3)
r1.metric("Current risk (model estimate)", f"{base_p:.1%}", category(base_p))
r2.metric(
    "What-if risk", f"{new_p:.1%}",
    f"{(new_p - base_p) * 100:+.1f} points",
    delta_color="inverse",
)
r3.metric("What-if category", category(new_p))

st.caption(
    "The current risk here comes from the final model trained on 80% of employees, so it can differ "
    "slightly from the cross-validated score in the profile above. The what-if shows what the model "
    "associates with these changes. It does not prove that the change would stop someone leaving."
)