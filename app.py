import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import shap

st.set_page_config(
    page_title="FarmerXential Dashboard",
    layout="wide"
)

st.title("FarmerXential — Agricultural Intelligence System")
st.caption("by Lalishank Holdings Limited")

df = pd.read_csv("farmerxential_farmer_priority_output.csv")
model = joblib.load("farmerxential_model.pkl")
features = joblib.load("farmerxential_features.pkl")
explainer = joblib.load("farmerxential_shap_explainer.pkl")

def get_farmer_problems(row):
    problems = []
    if row["yield_original"] < 1:
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
    if row["yield_original"] < 1:
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

zone_mapping = {
    0: "North Central",
    1: "North East",
    2: "North West",
    3: "South East",
    4: "South South",
    5: "South West"
}
df["zone"] = df["zone"].map(zone_mapping)

df = df.rename(columns={
    "yield_original": "Yield (kg/ha)",
    "land_size_original": "Land Size (m²)",
    "predicted_intervention_level": "Priority Level",
    "risk_score": "Risk Score (%)"
})

df["Risk Score (%)"] = (df["Risk Score (%)"] * 100).round(1)

df["Priority"] = df["Priority Level"].map({
    0: "🟢 Low",
    1: "🟡 Medium",
    2: "🔴 High"
})

st.sidebar.header("Filter Farmers")
selected_zone = st.sidebar.selectbox(
    "Select Zone",
    options=["All"] + list(df["zone"].unique())
)
if selected_zone != "All":
    df = df[df["zone"] == selected_zone]

st.write("### Farmer Priority Dataset")
st.dataframe(df[[
    "hhid",
    "Priority",
    "Risk Score (%)",
    "Yield (kg/ha)",
    "Land Size (m²)",
    "zone",
    "problems_detected",
    "recommended_actions"
]].head(20))

st.write("### Key Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Farmers", len(df))
with col2:
    high_priority_count = len(df[df["Priority Level"] == 2])
    st.metric("High Priority Farmers", high_priority_count)
with col3:
    st.metric("Average Risk Score", f"{df['Risk Score (%)'].mean():.1f}%")

st.write("### Intervention Distribution")
intervention_counts = df["Priority"].value_counts()
st.bar_chart(intervention_counts)

st.write("### Highest Priority Farmers")
high_priority_farmers = df[df["Priority Level"] == 2]
st.dataframe(
    high_priority_farmers[[
        "hhid",
        "Priority",
        "Risk Score (%)",
        "Yield (kg/ha)",
        "Land Size (m²)",
        "zone",
        "problems_detected",
        "recommended_actions"
    ]].sort_values("Risk Score (%)", ascending=False).head(20),
    use_container_width=True
)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Full Farmer Report",
    data=csv,
    file_name="farmerxential_farmer_report.csv",
    mime="text/csv"
)

st.write("### Zone Risk Summary")
zone_summary = df.groupby("zone")["Risk Score (%)"].mean().reset_index()
zone_summary.columns = ["Zone", "Average Risk Score (%)"]
zone_summary["Average Risk Score (%)"] = zone_summary["Average Risk Score (%)"].round(1)
st.dataframe(zone_summary, use_container_width=True)

fig = px.bar(
    zone_summary,
    x="Zone",
    y="Average Risk Score (%)",
    color="Average Risk Score (%)",
    title="Average Farmer Risk Score by Zone (%) — FarmerXential"
)
st.plotly_chart(fig, use_container_width=True)

st.write("### Predict New Farmer Intervention Need")
with st.form("farmer_prediction_form"):
    yield_value = st.number_input("Yield (kg/ha)", min_value=0.0, value=0.5)
    land_size = st.number_input("Land Size (m²)", min_value=0.0, value=1000.0)
    household_size = st.number_input("Household Size", min_value=1, value=5)
    household_max_education = st.number_input("Highest Education Level (0-26)", min_value=0, value=10)
    has_extension_access = st.selectbox("Has Extension Access?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    shock_level = st.selectbox("Shock Level", [0, 1, 2], format_func=lambda x: ["None", "Mild", "Severe"][x])
    received_assistance = st.selectbox("Received Assistance?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    used_fertilizer = st.selectbox("Used Fertilizer?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    received_credit = st.selectbox("Received Credit?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    zone = st.selectbox("Zone", [0, 1, 2, 3, 4, 5], format_func=lambda x: ["North Central", "North East", "North West", "South East", "South South", "South West"][x])
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
        "transport_cost": 0,
        "dependency_ratio": 0,
        "asset_score": 0,
        "postharvest_activity_score": 0,
        "crop_loss_risk_score": 0,
        "crop_diversity_score": 0,
        "digital_access_score": 0,
        "has_veterinary_access": 0,
        "market_access_score": 0,
        "is_rural": 1,
        "rainfall_anomaly": 0,
        "drought_risk": 0,
        "cultivates_crops": 1,
        "received_credit": received_credit,
        "head_gender": 1
    }])
    new_farmer = new_farmer[features]
    prediction = model.predict(new_farmer)[0]
    proba = model.predict_proba(new_farmer)[0][2]

    if prediction == 0:
        st.success(f"🟢 Low intervention need — Risk Score: {proba*100:.1f}%")
    elif prediction == 1:
        st.warning(f"🟡 Medium intervention need — Risk Score: {proba*100:.1f}%")
    else:
        st.error(f"🔴 High intervention need — Risk Score: {proba*100:.1f}%")

    st.write("#### Why this farmer was flagged:")
    shap_values_farmer = explainer.shap_values(new_farmer)
    shap_farmer_class2 = shap_values_farmer[:, :, 2][0]

    shap_explanation = pd.DataFrame({
        "Feature": new_farmer.columns.tolist(),
        "Impact": shap_farmer_class2
    }).sort_values("Impact", ascending=False)

    top_reasons = shap_explanation[shap_explanation["Impact"] > 0].head(5)
    for _, row in top_reasons.iterrows():
        st.write(f"🔴 **{row['Feature']}** is pushing this farmer towards high risk")

    protective = shap_explanation[shap_explanation["Impact"] < 0].tail(3)
    for _, row in protective.iterrows():
        st.write(f"🟢 **{row['Feature']}** is reducing this farmer's risk")