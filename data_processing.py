import pandas as pd


def load_harvest_data():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\secta3i_harvestw5.csv"
    df = pd.read_csv(path)
    return df


def load_extension_access_data():
    extension_df = pd.read_csv(
        r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Planting Wave 5\Agriculture\sect11l1_plantingw5.csv"
    )
    extension_df["has_extension_access"] = extension_df["s11l1q1"].astype(str).str.contains("YES", case=False, na=False).astype(int)
    extension_household = extension_df.groupby("hhid")["has_extension_access"].max().reset_index()
    return extension_household


def load_food_insecurity_data():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect7_harvestw5.csv"
    df = pd.read_csv(path)
    return df


def process_food_insecurity(df):
    df = df[['hhid', 's7q3']].copy()
    df['food_insecure'] = df['s7q3'].astype(str).str.strip().str.contains('YES').astype(int)
    return df[['hhid', 'food_insecure']]


def load_land_data():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\secta1_harvestw5.csv"
    df = pd.read_csv(path)
    return df


def calculate_yield(df_harvest, df_land):
    land = df_land[['hhid', 'plotid', 'sa1q1c']]
    df = df_harvest.merge(land, on=['hhid', 'plotid'], how='left')  # left not inner
    df = df.dropna(subset=['sa3iq9a', 'sa3iq9_conv', 'sa1q1c'])
    df['harvest_kg'] = df['sa3iq9a'] * df['sa3iq9_conv']
    df['yield'] = df['harvest_kg'] / df['sa1q1c'].replace(0, 1)
    df_household = df.groupby('hhid')['yield'].mean().reset_index()

    # Get ALL households from harvest file, fill missing yield with 0
    all_households = df_harvest[['hhid']].drop_duplicates()
    df_household = all_households.merge(df_household, on='hhid', how='left')
    df_household['yield'] = df_household['yield'].fillna(0)

    return df_household


def create_intervention_target(df):
    """
    Creates intervention priority classes using
    weighted vulnerability scoring.

    Higher weight = stronger vulnerability signal.

    Classes:
    0 = Low priority    (bottom 50%)
    1 = Medium priority (next 30%)
    2 = High priority   (top 20%)
    """

    df = df.copy()
    df["risk_score_raw"] = 0

    # ── HIGH WEIGHT signals (2 points) ──
    if "shock_level" in df.columns:
        df["risk_score_raw"] += (
            df["shock_level"] >= 2
        ).astype(int) * 2

    if "food_insecure" in df.columns:
        df["risk_score_raw"] += (
            df["food_insecure"] == 1
        ).astype(int) * 2

    if "asset_score" in df.columns:
        low_asset_threshold = df["asset_score"].quantile(0.33)
        df["risk_score_raw"] += (
            df["asset_score"] <= low_asset_threshold
        ).astype(int) * 2

    # ── MEDIUM WEIGHT signals (1.5 points) ──
    if "has_extension_access" in df.columns:
        df["risk_score_raw"] += (
            df["has_extension_access"] == 0
        ).astype(float) * 1.5

    if "received_credit" in df.columns:
        df["risk_score_raw"] += (
            df["received_credit"] == 0
        ).astype(float) * 1.5

    if "dependency_ratio" in df.columns:
        high_dependency = df["dependency_ratio"].quantile(0.67)
        df["risk_score_raw"] += (
            df["dependency_ratio"] >= high_dependency
        ).astype(float) * 1.5

    # ── STANDARD WEIGHT signals (1 point) ──
    if "used_fertilizer" in df.columns:
        df["risk_score_raw"] += (
            df["used_fertilizer"] == 0
        ).astype(int)

    if "market_access_score" in df.columns:
        low_market = df["market_access_score"].quantile(0.33)
        df["risk_score_raw"] += (
            df["market_access_score"] <= low_market
        ).astype(int)

    if "household_max_education" in df.columns:
        low_edu = df["household_max_education"].quantile(0.33)
        df["risk_score_raw"] += (
            df["household_max_education"] <= low_edu
        ).astype(int)

    if "crop_diversity_score" in df.columns:
        low_diversity = df["crop_diversity_score"].quantile(0.33)
        df["risk_score_raw"] += (
            df["crop_diversity_score"] <= low_diversity
        ).astype(int)

    # ── Convert to percentile classes ──
    df["risk_percentile"] = df["risk_score_raw"].rank(
        method="first",
        pct=True
    )

    df["intervention_level"] = 0

    df.loc[
        (df["risk_percentile"] > 0.50) &
        (df["risk_percentile"] <= 0.80),
        "intervention_level"
    ] = 1

    df.loc[
        df["risk_percentile"] > 0.80,
        "intervention_level"
    ] = 2

    print("\n=== RISK SCORE DISTRIBUTION ===")
    print(df["risk_score_raw"].describe())

    print("\n=== RISK SCORE VALUE COUNTS ===")
    print(df["risk_score_raw"].value_counts().sort_index())

    print("\n=== INTERVENTION CLASS DISTRIBUTION ===")
    print(df["intervention_level"].value_counts().sort_index())

    print(f"\nTotal households: {len(df)}")
    print(f"Low priority (Class 0): {(df['intervention_level'] == 0).sum()}")
    print(f"Medium priority (Class 1): {(df['intervention_level'] == 1).sum()}")
    print(f"High priority (Class 2): {(df['intervention_level'] == 2).sum()}")

    return df


def load_household_education_data():
    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect2_harvestw5.csv"
    df = pd.read_csv(path)
    print(df["s2q9"].isnull().sum())
    df["education_code"] = df["s2q9"].astype(str).str.extract(r"^(\d{1,2})").astype(float)
    household_education = df.groupby("hhid")["education_code"].max().reset_index(name="household_max_education")
    household_size = df.groupby("hhid").size().reset_index(name="household_size")
    household_education["household_max_education"] = household_education["household_max_education"].fillna(0)
    household_features = household_education.merge(household_size, on="hhid", how="left")
    print(household_features.isnull().sum())
    return household_features


def load_household_gender_data():
    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect1_harvestw5.csv"
    household = pd.read_csv(path)
    household = household[["hhid", "s1q2", "s1q3"]]
    household = household[
        household["s1q3"].astype(str).str.contains("HEAD", case=False, na=False)
    ]
    household["head_gender"] = household["s1q2"].map({
        "1. MALE": 1,
        "2. FEMALE": 0,
        1: 1,
        2: 0
    })
    household = household[["hhid", "head_gender"]]
    print(household["head_gender"].value_counts())
    print(household.isnull().sum())
    return household


def load_shock_data():
    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect12_harvestw5.csv"
    df = pd.read_csv(path)
    df = df[(df["s12q1"] == "1. YES") | (df["s12q1"] == 1)]
    shock_counts = df.groupby("hhid").size().reset_index(name="num_shocks")
    shock_counts["shock_level"] = 0
    shock_counts.loc[shock_counts["num_shocks"] == 1, "shock_level"] = 1
    shock_counts.loc[shock_counts["num_shocks"] >= 2, "shock_level"] = 2
    shock_counts = shock_counts[["hhid", "shock_level"]]
    print(shock_counts["shock_level"].value_counts())
    print(shock_counts.isnull().sum())
    return shock_counts


def load_assistance_data():
    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect11a_harvestw5.csv"
    df = pd.read_csv(path)
    df = df[["hhid", "s11q1a"]]
    df["received_assistance"] = df["s11q1a"].map({
        "1. YES": 1,
        "2. NO": 0,
        1: 1,
        2: 0
    })
    df = df[["hhid", "received_assistance"]]
    print(df["received_assistance"].value_counts())
    print(df.isnull().sum())
    return df


def load_fertilizer_data():
    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\secta11c2_harvestw5.csv"
    df = pd.read_csv(path)
    df = df[["hhid", "s11c2q5", "s11c2q11"]]
    inorganic = (df["s11c2q5"] == "1. YES") | (df["s11c2q5"] == 1)
    organic = (df["s11c2q11"] == "1. YES") | (df["s11c2q11"] == 1)
    df["used_fertilizer"] = 0
    df.loc[inorganic | organic, "used_fertilizer"] = 1
    fertilizer = df.groupby("hhid")["used_fertilizer"].max().reset_index()
    print(fertilizer["used_fertilizer"].value_counts())
    print(fertilizer.isnull().sum())
    return fertilizer


def load_plot_size_data():
    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Planting Wave 5\Agriculture\sect11a1_plantingw5.csv"
    df = pd.read_csv(path)
    df = df[["hhid", "s11aq3_number", "s11aq3_unit"]]
    df["land_size"] = 0.0
    df.loc[df["s11aq3_unit"] == "7. SQUARE METERS", "land_size"] = df["s11aq3_number"]
    df.loc[df["s11aq3_unit"] == "6. HECTARES", "land_size"] = df["s11aq3_number"] * 10000
    df.loc[df["s11aq3_unit"] == "5. ACRES", "land_size"] = df["s11aq3_number"] * 4046.86
    plot_size = df.groupby("hhid")["land_size"].sum().reset_index()
    plot_size["land_size"] = plot_size["land_size"].clip(upper=plot_size["land_size"].quantile(0.99))
    print(plot_size["land_size"].describe())
    print(plot_size.isnull().sum())
    return plot_size


def load_credit_access_data():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\secta12_harvestw5.csv"
    df = pd.read_csv(path)
    df = df[["hhid", "s12q20__3"]]
    df["received_credit"] = df["s12q20__3"].fillna(0)
    df["received_credit"] = df["received_credit"].astype(int)
    df = df[["hhid", "received_credit"]]
    df = df.groupby("hhid")["received_credit"].max().reset_index()
    print(df["received_credit"].value_counts())
    print(df.isnull().sum())
    return df


def load_zone_data():
    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect1_harvestw5.csv"
    df = pd.read_csv(path)
    df = df[["hhid", "zone"]]
    print(df.isnull().sum())
    df["zone"] = df["zone"].astype("category").cat.codes
    df = df.drop_duplicates(subset="hhid")
    return df


def load_dependency_ratio_data():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect1_harvestw5.csv"
    df = pd.read_csv(path)
    df = df[["hhid", "s1q6"]]
    df["age"] = pd.to_numeric(df["s1q6"], errors="coerce")
    df["dependent"] = ((df["age"] < 15) | (df["age"] > 64)).astype(int)
    df["working_age"] = ((df["age"] >= 15) & (df["age"] <= 64)).astype(int)
    dependency = df.groupby("hhid").agg(
        dependents=("dependent", "sum"),
        working_age_members=("working_age", "sum")
    ).reset_index()
    dependency["dependency_ratio"] = (
        dependency["dependents"] / dependency["working_age_members"].replace(0, 1)
    )
    dependency = dependency[["hhid", "dependency_ratio"]]
    print(dependency["dependency_ratio"].describe())
    print(dependency.isnull().sum())
    return dependency


def load_asset_data():
    asset_path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Planting Wave 5\Household\sect10_plantingw5.csv"
    assets = pd.read_csv(asset_path)
    owned_assets = assets[assets["s10q1a"] == "1. YES"].copy()
    owned_assets["s10q2"] = owned_assets["s10q2"].fillna(1)
    asset_features = owned_assets.groupby("hhid")["s10q2"].sum().reset_index()
    asset_features.rename(columns={"s10q2": "asset_score"}, inplace=True)
    asset_cap = asset_features["asset_score"].quantile(0.99)
    asset_features["asset_score"] = asset_features["asset_score"].clip(upper=asset_cap)
    return asset_features


def load_transport_cost_data():
    transport_path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\secta11c3q12_harvestw5.csv"
    transport = pd.read_csv(transport_path)
    transport_features = transport[["hhid", "s11c3q12"]].copy()
    transport_features.rename(columns={"s11c3q12": "transport_cost"}, inplace=True)
    transport_features["transport_cost"] = transport_features["transport_cost"].astype(str).str.extract(r"(\d+)")
    transport_features["transport_cost"] = pd.to_numeric(transport_features["transport_cost"], errors="coerce")
    transport_features["transport_cost"] = transport_features["transport_cost"].fillna(0)
    return transport_features


def load_market_access_data():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\secta11c3_harvestw5.csv"
    df = pd.read_csv(path)
    market = df[["hhid", "s11c3q8", "s11c3q10", "s11c3q11"]].copy()
    market["s11c3q10"] = market["s11c3q10"].astype(str).str.strip()
    market["market_interaction"] = market["s11c3q10"].str.contains("YES", case=False, na=False).astype(int)
    market["s11c3q11"] = pd.to_numeric(market["s11c3q11"], errors="coerce").fillna(0)
    market["distance_score"] = 0
    market.loc[market["s11c3q8"].astype(str).str.contains("LESS THAN 1", case=False, na=False), "distance_score"] = 3
    market["market_access_score"] = market["distance_score"] + market["market_interaction"]
    market_features = market.groupby("hhid")["market_access_score"].mean().reset_index()
    return market_features


def load_veterinary_access_data():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Planting Wave 5\Agriculture\sect11j_plantingw5.csv"
    df = pd.read_csv(path)
    vet = df[["hhid", "s11jq10"]].copy()
    print("\nVeterinary access missing values before cleaning:")
    print(vet.isnull().sum())
    vet["s11jq10"] = vet["s11jq10"].fillna("2. NO")
    vet["has_veterinary_access"] = vet["s11jq10"].astype(str).str.contains("YES", case=False, na=False).astype(int)
    vet_features = vet.groupby("hhid")["has_veterinary_access"].max().reset_index()
    print("\nVeterinary access missing values after cleaning:")
    print(vet_features.isnull().sum())
    return vet_features


def load_digital_access_data():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Planting Wave 5\Agriculture\sect12_plantingw5.csv"
    digital = pd.read_csv(path)
    digital_cols = ["hhid", "s12q33__1", "s12q33__3", "s12q33__5", "s12q33__8"]
    digital = digital[digital_cols].copy()
    usage_cols = ["s12q33__1", "s12q33__3", "s12q33__5", "s12q33__8"]
    digital[usage_cols] = digital[usage_cols].fillna(0)
    digital["digital_access_score"] = digital[usage_cols].sum(axis=1)
    digital_features = digital.groupby("hhid")["digital_access_score"].max().reset_index()
    print("\nDigital access missing values after cleaning:")
    print(digital_features.isnull().sum())
    return digital_features


def load_crop_diversity_data():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\secta3ii_harvestw5.csv"
    crops = pd.read_csv(path)
    crops = crops[["hhid", "cropcode"]].copy()
    crops = crops.dropna(subset=["hhid", "cropcode"])
    crop_diversity = crops.groupby("hhid")["cropcode"].nunique().reset_index()
    crop_diversity.rename(columns={"cropcode": "crop_diversity_score"}, inplace=True)
    print("\nCrop diversity statistics:")
    print(crop_diversity["crop_diversity_score"].describe())
    return crop_diversity


def load_postharvest_activity_data():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\sectaphl1_harvestw5.csv"
    postharvest = pd.read_csv(path)
    activity_cols = ["phl_q1", "phl_q2", "phl_q4__1", "phl_q4__2", "phl_q4__3", "phl_q4__4", "phl_q4__5", "phl_q4__6"]
    available_cols = ["hhid"] + [col for col in activity_cols if col in postharvest.columns]
    postharvest = postharvest[available_cols].copy()
    postharvest = postharvest.fillna(0)
    for col in activity_cols:
        if col in postharvest.columns:
            postharvest[col] = pd.to_numeric(postharvest[col], errors="coerce").fillna(0)
    postharvest["postharvest_activity_score"] = postharvest[[c for c in activity_cols if c in postharvest.columns]].sum(axis=1)
    postharvest_features = postharvest.groupby("hhid")["postharvest_activity_score"].max().reset_index()
    print("\nPost-harvest activity statistics:")
    print(postharvest_features["postharvest_activity_score"].describe())
    return postharvest_features


def load_crop_loss_data():
    loss_path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\secta3ii_harvestw5.csv"
    loss_data = pd.read_csv(loss_path)
    loss_data["sa3iiq20"] = pd.to_numeric(loss_data["sa3iiq20"], errors="coerce").fillna(0)
    crop_loss_features = loss_data.groupby("hhid")["sa3iiq20"].sum().reset_index()
    crop_loss_features = crop_loss_features.rename(columns={"sa3iiq20": "crop_loss_risk_score"})
    print("\nCrop loss statistics:")
    print(crop_loss_features["crop_loss_risk_score"].describe())
    return crop_loss_features


def load_livestock_data():
    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Planting Wave 5\Agriculture\sect11a_plantingw5.csv"
    df = pd.read_csv(path)
    print("\nMissing values before cleaning:")
    print(df[["ag3", "ag4"]].isnull().sum())
    df["ag3"] = df["ag3"].fillna("2. NO")
    df["ag4"] = df["ag4"].fillna("2. NO")
    df["has_livestock"] = (
        (df["ag3"].astype(str).str.contains("YES", case=False, na=False)) |
        (df["ag4"].astype(str).str.contains("YES", case=False, na=False))
    ).astype(int)
    livestock = df.groupby("hhid")["has_livestock"].max().reset_index()
    print("\nLivestock distribution:")
    print(livestock["has_livestock"].value_counts())
    print("\nMissing values in final output:")
    print(livestock.isnull().sum())
    return livestock

def load_household_ag_features():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\secta_harvestw5.csv"
    df = pd.read_csv(path)
    df = df[["hhid", "sector", "ag1"]].copy()
    df["is_rural"] = df["sector"].astype(str).str.contains("2", na=False).astype(int)
    df["cultivates_crops"] = (
        df["ag1"].astype(str).str.contains("YES", case=False, na=False)
    ).astype(int)
    df = df[["hhid", "is_rural", "cultivates_crops"]]
    df = df.drop_duplicates(subset="hhid")
    print("\nAg features distribution:")
    print(df[["is_rural", "cultivates_crops"]].value_counts())
    return df


def load_rainfall_features():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect1_harvestw5.csv"
    df = pd.read_csv(path)
    df = df[["hhid", "state"]].drop_duplicates(subset="hhid")

    # 2023 Nigeria growing season rainfall anomaly by state
    # Negative = below normal rainfall (drought risk)
    # Positive = above normal rainfall (flood risk)
    # Source: CHIRPS 2023 Nigeria analysis
    rainfall_lookup = {
        "1. Abia": 0.3,
        "2. Adamawa": -0.8,
        "3. Akwa Ibom": 0.4,
        "4. Anambra": 0.2,
        "5. Bauchi": -0.9,
        "6. Bayelsa": 0.5,
        "7. Benue": -0.3,
        "8. Borno": -1.2,
        "9. Cross River": 0.3,
        "10. Delta": 0.4,
        "11. Ebonyi": 0.1,
        "12. Edo": 0.3,
        "13. Ekiti": -0.1,
        "14. Enugu": 0.0,
        "15. Gombe": -0.7,
        "16. Imo": 0.2,
        "17. Jigawa": -0.6,
        "18. Kaduna": -0.4,
        "19. Kano": -0.5,
        "20. Katsina": -0.8,
        "21. Kebbi": -0.6,
        "22. Kogi": -0.2,
        "23. Kwara": -0.3,
        "24. Lagos": 0.5,
        "25. Nasarawa": -0.4,
        "26. Niger": -0.5,
        "27. Ogun": 0.1,
        "28. Ondo": 0.2,
        "29. Osun": 0.0,
        "30. Oyo": -0.1,
        "31. Plateau": -0.3,
        "32. Rivers": 0.6,
        "33. Sokoto": -0.9,
        "34. Taraba": -0.5,
        "35. Yobe": -1.0,
        "36. Zamfara": -0.7,
        "37. FCT": -0.2
    }

    df["rainfall_anomaly"] = df["state"].map(rainfall_lookup)
    df["rainfall_anomaly"] = df["rainfall_anomaly"].fillna(0)

    # Drought risk flag — states with severe below normal rainfall
    df["drought_risk"] = (df["rainfall_anomaly"] < -0.5).astype(int)

    print("\nRainfall anomaly distribution:")
    print(df["rainfall_anomaly"].describe())
    print("\nDrought risk distribution:")
    print(df["drought_risk"].value_counts())

    return df[["hhid", "rainfall_anomaly", "drought_risk"]]