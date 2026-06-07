# import pandas as pd
# from sklearn.model_selection import train_test_split
# import seaborn as sns
# import matplotlib.pyplot as plt
# import joblib
# pd.set_option('display.max_columns', None)
#
# from xgboost import XGBClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#
# from data_processing import (
#     load_harvest_data,
#     load_land_data,
#     calculate_yield,
#     load_plot_size_data,
#     load_zone_data,
#     load_extension_access_data,
#     load_household_education_data,
#     load_household_gender_data,
#     load_food_insecurity_data,
#     process_food_insecurity,
#     load_assistance_data,
#     load_credit_access_data,
#     load_asset_data,
#     load_rainfall_features,
#     load_crop_diversity_data,
#     load_digital_access_data,
#     load_dependency_ratio_data,
#     load_fertilizer_data,
#     load_household_ag_features,
#     load_postharvest_activity_data,
#     load_crop_loss_data,
#     load_veterinary_access_data,
#     load_transport_cost_data,
#     load_market_access_data,
#     load_shock_data,
#     load_livestock_data,
#     create_intervention_target
# )
#
# df_harvest = load_harvest_data()
# df_land = load_land_data()
# df_plot_size = load_plot_size_data()
# df_food_raw = load_food_insecurity_data()
# df_transport = load_transport_cost_data()
# df_assistance = load_assistance_data()
# df_shock = load_shock_data()
# df_market_access = load_market_access_data()
# veterinary_features = load_veterinary_access_data()
# df_fertilizer = load_fertilizer_data()
# crop_loss_features = load_crop_loss_data()
# digital_features = load_digital_access_data()
# crop_diversity = load_crop_diversity_data()
# df_livestock = load_livestock_data()
# df_ag_features = load_household_ag_features()
# postharvest_features = load_postharvest_activity_data()
# df_dependency = load_dependency_ratio_data()
# df_rainfall = load_rainfall_features()
# df_credit = load_credit_access_data()
# df_zone = load_zone_data()
# df_assets = load_asset_data()
#
# df_yield = calculate_yield(df_harvest, df_land)
# df_food = process_food_insecurity(df_food_raw)
#
# df_final = df_yield.merge(df_food, on='hhid', how='left')
# df_final = create_intervention_target(df_final)
#
# df_extension = load_extension_access_data()
# df_education = load_household_education_data()
# df_gender = load_household_gender_data()
#
# df_final = df_final.merge(veterinary_features, on="hhid", how="left")
# df_final = df_final.merge(df_extension, on="hhid", how="left")
# df_final = df_final.merge(digital_features, on="hhid", how="left")
# df_final = df_final.merge(df_education, on="hhid", how="left")
# df_final = df_final.merge(df_gender, on="hhid", how="left")
# df_final = df_final.merge(df_transport, on="hhid", how="left")
# df_final = df_final.merge(df_market_access, on="hhid", how="left")
# df_final = df_final.merge(df_shock, on="hhid", how="left")
# df_final = df_final.merge(df_assets, on="hhid", how="left")
# df_final = df_final.merge(crop_diversity, on="hhid", how="left")
# df_final = df_final.merge(df_fertilizer, on="hhid", how="left")
# df_final = df_final.merge(df_assistance, on="hhid", how="left")
# df_final = df_final.merge(df_dependency, on="hhid", how="left")
# df_final = df_final.merge(postharvest_features, on="hhid", how="left")
# df_final = df_final.merge(crop_loss_features, on="hhid", how="left")
# df_final = df_final.merge(df_plot_size, on="hhid", how="left")
# df_final = df_final.merge(df_credit, on="hhid", how="left")
# df_final = df_final.merge(df_zone, on="hhid", how="left")
# df_final = df_final.merge(df_ag_features, on="hhid", how="left")
# df_final = df_final.merge(df_rainfall, on="hhid", how="left")
#
# df_final["rainfall_anomaly"] = df_final["rainfall_anomaly"].fillna(0)
# df_final["drought_risk"] = df_final["drought_risk"].fillna(0)
# df_final["is_rural"] = df_final["is_rural"].fillna(1)
# df_final["cultivates_crops"] = df_final["cultivates_crops"].fillna(0)
# df_final["has_extension_access"] = df_final["has_extension_access"].fillna(0)
# df_final["land_size"] = df_final["land_size"].fillna(0)
# df_final["household_max_education"] = df_final["household_max_education"].fillna(0)
# df_final["digital_access_score"] = df_final["digital_access_score"].fillna(0)
# df_final["crop_diversity_score"] = df_final["crop_diversity_score"].fillna(0)
# df_final["has_extension_access"] = df_final["has_extension_access"].astype(int)
# df_final["has_veterinary_access"] = df_final["has_veterinary_access"].fillna(0)
# df_final["postharvest_activity_score"] = df_final["postharvest_activity_score"].fillna(0)
# df_final["crop_loss_risk_score"] = df_final["crop_loss_risk_score"].fillna(0)
# df_final["market_access_score"] = df_final["market_access_score"].fillna(0)
# df_final["received_assistance"] = df_final["received_assistance"].fillna(0)
# df_final["transport_cost"] = df_final["transport_cost"].fillna(0)
# df_final["shock_level"] = df_final["shock_level"].fillna(0)
# df_final["asset_score"] = df_final["asset_score"].fillna(0)
# df_final["dependency_ratio"] = df_final["dependency_ratio"].fillna(0)
# df_final["received_credit"] = df_final["received_credit"].fillna(0)
# df_final["food_insecure"] = df_final["food_insecure"].fillna(0)
# df_final["head_gender"] = df_final["head_gender"].fillna(1)
# df_final["used_fertilizer"] = df_final["used_fertilizer"].fillna(0)
# df_final["household_size"] = df_final["household_size"].fillna(0)
# df_final["zone"] = df_final["zone"].fillna(0)
#
# print(df_final.shape)
# print("\nIntervention levels:")
# print(df_final["intervention_level"].value_counts())
#
# features = [
#     "yield",
#     "has_extension_access",
#     "household_max_education",
#     "shock_level",
#     "received_assistance",
#     "used_fertilizer",
#     "land_size",
#     "household_size",
#     "zone",
#     "transport_cost",
#     "dependency_ratio",
#     "asset_score",
#     "postharvest_activity_score",
#     "crop_loss_risk_score",
#     "crop_diversity_score",
#     "digital_access_score",
#     "has_veterinary_access",
#     "market_access_score",
#     "is_rural",
#     "rainfall_anomaly",
#     "drought_risk",
#     "cultivates_crops",
#     "received_credit",
#     "head_gender"
# ]
#
# target = "intervention_level"
#
# yield_upper = df_final["yield"].quantile(0.99)
# df_final["yield"] = df_final["yield"].clip(upper=yield_upper)
#
# transport_upper = df_final["transport_cost"].quantile(0.99)
# df_final["transport_cost"] = df_final["transport_cost"].clip(upper=transport_upper)
#
# continuous_outlier_features = [
#     "land_size",
#     "transport_cost",
#     "asset_score",
#     "postharvest_activity_score",
#     "crop_loss_risk_score",
#     "household_size",
#     "dependency_ratio"
# ]
#
# for col in continuous_outlier_features:
#     upper_limit = df_final[col].quantile(0.99)
#     df_final[col] = df_final[col].clip(upper=upper_limit)
#
# from sklearn.preprocessing import StandardScaler
#
# scale_columns = [
#     "land_size",
#     "yield",
#     "transport_cost",
#     "asset_score",
#     "postharvest_activity_score",
#     "crop_loss_risk_score",
#     "household_size",
#     "dependency_ratio"
# ]
#
# df_final["yield_original"] = df_final["yield"]
# df_final["land_size_original"] = df_final["land_size"]
#
# scaler = StandardScaler()
# df_final[scale_columns] = scaler.fit_transform(df_final[scale_columns])
#
# print("\nFEATURE SCALING COMPLETED")
#
# X = df_final[features]
# y = df_final[target]
#
# print("\nMissing values per feature:")
# print(X.isnull().sum())
#
# print("\nTarget distribution:")
# print(y.value_counts())
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# from imblearn.over_sampling import SMOTE
#
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)
#
# print("\nClass distribution after SMOTE:")
# print(pd.Series(y_train).value_counts())
#
# print("\nTEST CLASS DISTRIBUTION")
# print(y_test.value_counts())
#
# from sklearn.ensemble import VotingClassifier
# from sklearn.linear_model import LogisticRegression
#
# # ==============================
# # BEST PARAMS FROM GRIDSEARCH
# # ==============================
# rf_model = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=16,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     class_weight="balanced",
#     random_state=42
# )
#
# xgb_model = XGBClassifier(
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.05,
#     scale_pos_weight=1.5,
#     random_state=42,
#     eval_metric="mlogloss",
#     use_label_encoder=False
# )
#
# lr_model = LogisticRegression(
#     max_iter=1000,
#     class_weight="balanced",
#     random_state=42
# )
#
# model = VotingClassifier(
#     estimators=[
#         ("rf", rf_model),
#         ("xgb", xgb_model),
#         ("lr", lr_model)
#     ],
#     voting="soft"
# )
#
# model.fit(X_train, y_train)
# print("Model training completed")
#
# y_pred = model.predict(X_test)
# print(y_pred[:10])
#
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
#
# from sklearn.metrics import roc_auc_score, roc_curve, auc
# from sklearn.preprocessing import label_binarize
#
# y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
# y_proba = model.predict_proba(X_test)
#
# roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
# print(f"\nROC AUC Score: {roc_auc:.4f}")
# print("(0.5 = random guessing, 1.0 = perfect)")
#
# fig, ax = plt.subplots(figsize=(8, 6))
# colors = ['#639922', '#EF9F27', '#E24B4A']
# class_names = ['Low Priority', 'Medium Priority', 'High Priority']
#
# for i in range(3):
#     fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
#     roc_auc_class = auc(fpr, tpr)
#     ax.plot(fpr, tpr, color=colors[i], lw=2,
#             label=f'{class_names[i]} (AUC = {roc_auc_class:.2f})')
#
# ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random guessing (AUC = 0.50)')
# ax.set_xlim([0.0, 1.0])
# ax.set_ylim([0.0, 1.05])
# ax.set_xlabel('False Positive Rate', fontsize=12)
# ax.set_ylabel('True Positive Rate', fontsize=12)
# ax.set_title('FarmerXential ROC Curve — Model vs Random Guessing', fontsize=13)
# ax.legend(loc="lower right")
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("farmerxential_roc_curve.png", dpi=150)
# plt.show()
# print("\nROC curve saved as farmerxential_roc_curve.png")
#
# feature_importance = pd.DataFrame({
#     "feature": features,
#     "importance": model.estimators_[0].feature_importances_
# }).sort_values(by="importance", ascending=False)
# print(feature_importance)
#
# # ==============================
# # PRODUCT OUTPUT: FARMER RISK SCORE
# # ==============================
# df_output = df_final.copy()
# df_output["predicted_intervention_level"] = model.predict(df_final[features])
# prediction_probabilities = model.predict_proba(df_final[features])
# df_output["risk_score"] = prediction_probabilities[:, 2]
#
# df_output = df_output[[
#     "hhid",
#     "intervention_level",
#     "predicted_intervention_level",
#     "risk_score",
#     "yield_original",
#     "land_size_original",
#     "household_size",
#     "used_fertilizer",
#     "household_max_education",
#     "has_extension_access",
#     "shock_level",
#     "asset_score",
#     "postharvest_activity_score",
#     "crop_loss_risk_score",
#     "digital_access_score",
#     "market_access_score",
#     "transport_cost",
#     "received_credit",
#     "zone"
# ]]
#
# print("\nHighest priority farmers:")
# print(df_output.sort_values(by="risk_score", ascending=False).head(20))
#
# # ==============================
# # EXPLANATION ENGINE
# # ==============================
# high_priority = df_output[df_output["predicted_intervention_level"] == 2]
#
# print("\nWhy farmers were flagged as HIGH PRIORITY:\n")
# for index, row in high_priority.head(10).iterrows():
#     reasons = []
#     if row["yield_original"] < 1:
#         reasons.append("Low yield")
#     if row["has_extension_access"] == 0:
#         reasons.append("No extension access")
#     if row["shock_level"] > 0:
#         reasons.append("Experienced agricultural shock")
#     if row["received_credit"] == 0:
#         reasons.append("No access to credit")
#     if row["household_max_education"] <= 1:
#         reasons.append("Low household education")
#     print(f"Farmer {row['hhid']} flagged because:")
#     for reason in reasons:
#         print("-", reason)
#     print("-------------------")
#
# print("\nRecommended interventions:\n")
# for index, row in high_priority.head(10).iterrows():
#     recommendations = []
#     if row["yield_original"] < 1:
#         recommendations.append("Provide yield improvement support")
#     if row["has_extension_access"] == 0:
#         recommendations.append("Assign extension officer")
#     if row["shock_level"] > 0:
#         recommendations.append("Provide emergency agricultural support")
#     if row["received_credit"] == 0:
#         recommendations.append("Provide access to agricultural credit")
#     if row["used_fertilizer"] == 0:
#         recommendations.append("Provide fertilizer/input support")
#     print(f"Farmer {row['hhid']} recommendations:")
#     for rec in recommendations:
#         print("-", rec)
#     print("-------------------")
#
# df_output.to_csv("farmerxential_farmer_priority_output.csv", index=False)
# print("\nFarmerXential output saved successfully.")
#
# joblib.dump(model, "farmerxential_model.pkl")
# joblib.dump(features, "farmerxential_features.pkl")
# print("FarmerXential model and features saved successfully.")
#
# # ==============================
# # SHAP EXPLAINABILITY
# # ==============================
# import shap
#
# print("\nCalculating SHAP values...")
#
# explainer = shap.TreeExplainer(model.estimators_[0])
# shap_values = explainer.shap_values(X_test)
# shap_values_class2 = shap_values[:, :, 2]
#
# shap_importance = pd.DataFrame({
#     "feature": X_test.columns.tolist(),
#     "shap_importance": abs(shap_values_class2).mean(axis=0)
# }).sort_values("shap_importance", ascending=False)
#
# print("\nSHAP Feature Importance for High Priority Farmers:")
# print(shap_importance)
#
# joblib.dump(explainer, "farmerxential_shap_explainer.pkl")
# print("\nSHAP explainer saved successfully.")