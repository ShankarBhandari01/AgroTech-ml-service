import pandas as pd
from sklearn.model_selection import train_test_split

import joblib


from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data_processing import (
    load_harvest_data,
    load_land_data,
    calculate_yield,
    load_plot_size_data,
    load_zone_data,
    load_extension_access_data,
    load_household_education_data,
    load_household_gender_data,
    load_food_insecurity_data,
    process_food_insecurity,
    load_assistance_data,
    load_credit_access_data,
    load_asset_data,
    load_dependency_ratio_data,
    load_fertilizer_data,
    load_transport_cost_data,
    load_shock_data,
    create_intervention_target
)

df_harvest = load_harvest_data()
df_land = load_land_data()
df_plot_size = load_plot_size_data()
df_food_raw = load_food_insecurity_data()
df_transport = load_transport_cost_data()
df_assistance = load_assistance_data()
df_shock = load_shock_data()
df_fertilizer = load_fertilizer_data()
df_dependency = load_dependency_ratio_data()
df_credit = load_credit_access_data()
df_zone = load_zone_data()
df_assets = load_asset_data()

df_yield = calculate_yield(df_harvest, df_land)

df_food = process_food_insecurity(df_food_raw)

df_final = df_yield.merge(df_food, on='hhid', how='inner')

df_final = create_intervention_target(df_final)

df_extension = load_extension_access_data()


df_education = load_household_education_data()

df_gender = load_household_gender_data()

df_final = df_final.merge(df_extension, on="hhid", how="left")

df_final = df_final.merge(df_education, on="hhid", how="left")

df_final = df_final.merge(df_gender, on="hhid", how="left")

df_final = df_final.merge(df_transport, on="hhid", how="left")

df_final = df_final.merge(df_shock, on="hhid", how="left")

df_final = df_final.merge(df_assets, on="hhid", how="left")

df_final = df_final.merge(df_fertilizer, on="hhid", how="left")

df_final = df_final.merge(df_assistance, on="hhid", how="left")

df_final = df_final.merge(
    df_dependency,
    on="hhid",
    how="left"
)

df_final = df_final.merge(df_plot_size, on="hhid", how="left")

df_final = df_final.merge(df_credit, on="hhid", how="left")

df_final = df_final.merge(df_zone, on="hhid", how="left")

print(df_final.head())

print(df_final[[
    "household_max_education",
    "household_size"
]].isnull().sum())

df_final["has_extension_access"] = df_final["has_extension_access"].fillna(0)

df_final["land_size"] = df_final["land_size"].fillna(0)

df_final["household_max_education"] = df_final["household_max_education"].fillna(0)

df_final["has_extension_access"] = df_final["has_extension_access"].astype(int)

df_final["received_assistance"] = df_final["received_assistance"].fillna(0)

df_final["transport_cost"] = df_final["transport_cost"].fillna(0)

df_final["shock_level"] = df_final["shock_level"].fillna(0)

df_final["asset_score"] = df_final["asset_score"].fillna(0)

df_final["dependency_ratio"] = df_final["dependency_ratio"].fillna(0)

df_final["received_credit"] = df_final["received_credit"].fillna(0)

print(df_final["has_extension_access"].value_counts())
print(df_final.shape)
print("\nFinal dataset:")
print(df_final.head())
print("\nIntervention levels:")
print(df_final["intervention_level"].value_counts())

features = [
    "yield",
    "has_extension_access",
    "household_max_education",
    "shock_level",
    "received_assistance",
    "used_fertilizer",
    "land_size",
    "household_size",
    "zone",
    "transport_cost",
    "dependency_ratio",
    "asset_score",
    "received_credit",
    "head_gender"
]
target = "intervention_level"

X = df_final[features]

y = df_final[target]

print(X.head())

print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed")

y_pred = model.predict(X_test)

print(y_pred[:10])

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print(feature_importance)

# ==============================
# PRODUCT OUTPUT: FARMER RISK SCORE
# ==============================

df_output = df_final.copy()

df_output["predicted_intervention_level"] = model.predict(df_final[features])

prediction_probabilities = model.predict_proba(df_final[features])

df_output["risk_score"] = df_output["predicted_intervention_level"] / 2

df_output = df_output[[
    "hhid",
    "intervention_level",
    "predicted_intervention_level",
    "risk_score",
    "yield",
    "land_size",
    "household_size",
    "used_fertilizer",
    "household_max_education",
    "has_extension_access",
    "shock_level",
    "asset_score",
    "transport_cost",
    "received_credit",
    "zone"
]]

print("\nHighest priority farmers:")
print(df_output.sort_values(by="risk_score", ascending=False).head(20))

print("\nAgroReach farmer priority output:")
print(df_output.head(20))

# ==============================
# SIMPLE EXPLANATION ENGINE
# ==============================

high_priority = df_output[df_output["predicted_intervention_level"] == 2]

print("\nWhy farmers were flagged as HIGH PRIORITY:\n")

for index, row in high_priority.head(10).iterrows():

    reasons = []

    if row["yield"] < 1:
        reasons.append("Low yield")

    if row["has_extension_access"] == 0:
        reasons.append("No extension access")

    if row["shock_level"] > 0:
        reasons.append("Experienced agricultural shock")

    if row["received_credit"] == 0:
        reasons.append("No access to credit")

    if row["household_max_education"] <= 1:
        reasons.append("Low household education")

    print(f"Farmer {row['hhid']} flagged because:")

    for reason in reasons:
        print("-", reason)

    print("-------------------")

# ==============================
# INTERVENTION RECOMMENDATION ENGINE
# ==============================

print("\nRecommended interventions:\n")

for index, row in high_priority.head(10).iterrows():

    recommendations = []

    if row["yield"] < 1:
        recommendations.append("Provide yield improvement support")

    if row["has_extension_access"] == 0:
        recommendations.append("Assign extension officer")

    if row["shock_level"] > 0:
        recommendations.append("Provide emergency agricultural support")

    if row["received_credit"] == 0:
        recommendations.append("Provide access to agricultural credit")

    if row["used_fertilizer"] == 0:
        recommendations.append("Provide fertilizer/input support")

    print(f"Farmer {row['hhid']} recommendations:")

    for rec in recommendations:
        print("-", rec)

    print("-------------------")

df_output.to_csv("agroreach_farmer_priority_output.csv", index=False)

print("\nAgroReach output saved successfully as agroreach_farmer_priority_output.csv")

joblib.dump(model, "agroreach_model.pkl")
joblib.dump(features, "agroreach_features.pkl")

print("AgroReach model and features saved successfully.")


market_path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\secta11c3_harvestw5.csv"

market_data = pd.read_csv(market_path)

print(market_data.columns)

print(market_data[
    [
        "hhid",
        "s11c3q8",
        "s11c3q10",
        "s11c3q11"
    ]
].head(20))