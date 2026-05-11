import streamlit as st
import pandas as pd

import joblib

st.set_page_config(
    page_title="AgroReach Dashboard",
    layout="wide"
)

st.title("AgroReach Agricultural Intelligence System")

df = pd.read_csv("agroreach_farmer_priority_output.csv")

model = joblib.load("agroreach_model.pkl")
features = joblib.load("agroreach_features.pkl")

def get_farmer_problems(row):
    problems = []

    if row["yield"] < 1:
        problems.append("Low yield")

    if row["has_extension_access"] == 0:
        problems.append("No extension access")

    if row["shock_level"] > 0:
        problems.append("Shock affected household")

    if row["received_credit"] == 0:
        problems.append("No credit access")

    if row["used_fertilizer"] == 0:
        problems.append("No fertilizer/input use")

    return ", ".join(problems)


def get_farmer_recommendations(row):
    recommendations = []

    if row["yield"] < 1:
        recommendations.append("Yield improvement support")

    if row["has_extension_access"] == 0:
        recommendations.append("Assign extension officer")

    if row["shock_level"] > 0:
        recommendations.append("Emergency agricultural support")

    if row["received_credit"] == 0:
        recommendations.append("Agricultural credit access")

    if row["used_fertilizer"] == 0:
        recommendations.append("Fertilizer/input support")

    return ", ".join(recommendations)


df["problems_detected"] = df.apply(get_farmer_problems, axis=1)
df["recommended_actions"] = df.apply(get_farmer_recommendations, axis=1)

st.sidebar.header("Filter Farmers")

selected_zone = st.sidebar.selectbox(
    "Select Zone",
    options=["All"] + list(df["zone"].unique())
)

if selected_zone != "All":
    df = df[df["zone"] == selected_zone]

st.write("### Farmer Priority Dataset")
st.dataframe(df.head(20))

st.write("### Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Farmers", len(df))

with col2:
    high_priority = df[df["predicted_intervention_level"] == 2]
    st.metric("High Priority Farmers", len(high_priority))

with col3:
    st.metric("Average Risk Score", round(df["risk_score"].mean(), 2))

st.write("### Intervention Distribution")

intervention_counts = df["predicted_intervention_level"].value_counts()

st.bar_chart(intervention_counts)

st.write("### Highest Priority Farmers")

st.write("### Highest Priority Farmers with Problems and Recommendations")

high_priority_farmers = df[df["predicted_intervention_level"] == 2]

st.dataframe(
    high_priority_farmers[[
        "hhid",
        "risk_score",
        "yield",
        "land_size",
        "household_size",
        "zone",
        "problems_detected",
        "recommended_actions"
    ]].head(20),
    use_container_width=True
)

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Full Farmer Report",
    data=csv,
    file_name="agroreach_farmer_report.csv",
    mime="text/csv"
)

st.write("### Input and Credit Summary")
st.write(df["used_fertilizer"].value_counts())
st.write(df["received_credit"].value_counts())

st.write("### Zone Risk Summary")

zone_mapping = {
    0: "North Central",
    1: "North East",
    2: "North West",
    3: "South East",
    4: "South South",
    5: "South West"
}

zone_summary = df.groupby("zone")["predicted_intervention_level"].mean().reset_index()

zone_summary.columns = ["Zone", "Average Risk Level"]

zone_summary["Zone"] = zone_summary["Zone"].map(zone_mapping)

st.dataframe(zone_summary, use_container_width=True)

import plotly.express as px

fig = px.bar(
    zone_summary,
    x="Zone",
    y="Average Risk Level",
    color="Average Risk Level",
    title="Average Farmer Risk by Zone"
)

st.plotly_chart(fig, use_container_width=True)

st.write("### Predict New Farmer Intervention Need")

with st.form("farmer_prediction_form"):

    yield_value = st.number_input("Yield", min_value=0.0, value=0.5)
    land_size = st.number_input("Land Size", min_value=0.0, value=1000.0)
    household_size = st.number_input("Household Size", min_value=1, value=5)
    household_max_education = st.number_input("Household Max Education", min_value=0, value=20)

    has_extension_access = st.selectbox("Has Extension Access?", [0, 1])
    shock_level = st.selectbox("Shock Level", [0, 1, 2])
    received_assistance = st.selectbox("Received Assistance?", [0, 1])
    used_fertilizer = st.selectbox("Used Fertilizer?", [0, 1])
    received_credit = st.selectbox("Received Credit?", [0, 1])
    zone = st.selectbox("Zone", [0, 1, 2, 3, 4, 5])

    submitted = st.form_submit_button("Predict Intervention Need")

if submitted:
    new_farmer = pd.DataFrame([{
        "yield": yield_value,
        "has_extension_access": has_extension_access,
        "household_max_education": household_max_education,
        "shock_level": shock_level,
        "received_assistance": received_assistance,
        "used_fertilizer": used_fertilizer,
        "land_size": land_size,
        "household_size": household_size,
        "zone": zone,
        "received_credit": received_credit,
        "head_gender": 1
    }])

    new_farmer = new_farmer[features]

    prediction = model.predict(new_farmer)[0]

    if prediction == 0:
        st.success("Low intervention need")
    elif prediction == 1:
        st.warning("Medium intervention need")
    else:
        st.error("High intervention need")