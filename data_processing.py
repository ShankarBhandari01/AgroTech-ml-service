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

    # Convert to binary
    df['food_insecure'] = df['s7q3'].astype(str).str.strip().str.contains('YES').astype(int)

    return df[['hhid', 'food_insecure']]


def load_land_data():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\secta1_harvestw5.csv"

    df = pd.read_csv(path)

    return df

extension_df = pd.read_csv(
    r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Planting Wave 5\Agriculture\sect11l1_plantingw5.csv"
)

print(extension_df.shape)
print(extension_df.columns)

print(extension_df["s11l1q1"].isnull().sum())

extension_df["has_extension_access"] = extension_df["s11l1q1"].astype(str).str.contains("YES", case=False, na=False).astype(int)

extension_household = extension_df.groupby("hhid")["has_extension_access"].max().reset_index()
print(extension_df["hhid"].nunique())
print(extension_df.loc[extension_df["s11l1q1"].str.contains("YES", case=False, na=False)].head())

print(extension_household.shape)

print(extension_household.shape)

print(extension_household["has_extension_access"].value_counts())

print(extension_df["s11l1q1"].value_counts())

def calculate_yield(df_harvest, df_land):
    # Example assumption (we will refine later)

    # Select land size column (we may adjust after checking)
    land = df_land[['hhid', 'plotid', 'sa1q1c']]  # ← this may change after inspection

    # Merge
    df = df_harvest.merge(land, on=['hhid', 'plotid'], how='inner')

    df = df.dropna(subset=['sa3iq9a', 'sa3iq9_conv', 'sa1q1c'])
    # Calculate yield
    df['harvest_kg'] = df['sa3iq9a'] * df['sa3iq9_conv']
    df['yield'] = df['harvest_kg'] / df['sa1q1c']

    df_household = df.groupby('hhid')['yield'].mean().reset_index()

    return df_household

def create_intervention_target(df):

    low_yield = df['yield'] < df['yield'].quantile(0.25)
    food_insecure = df['food_insecure'] == 1

    df['intervention_level'] = 0

    # HIGH PRIORITY (both problems)
    df.loc[low_yield & food_insecure, 'intervention_level'] = 2

    # MEDIUM PRIORITY (one problem)
    df.loc[(low_yield | food_insecure) & (df['intervention_level'] == 0), 'intervention_level'] = 1

    return df



def load_household_education_data():

    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect2_harvestw5.csv"

    df = pd.read_csv(path)

    print(df["s2q9"].isnull().sum())


    df["education_code"] = df["s2q9"].astype(str).str.extract(r"^(\d{1,2})").astype(float)

    household_education = df.groupby("hhid")["education_code"].max().reset_index(name="household_max_education")


    household_size = df.groupby("hhid").size().reset_index(name="household_size")

    print(household_size.head())

    print(household_education.head())

    household_education["household_max_education"] = household_education["household_max_education"].fillna(0)

    print(household_education.isnull().sum())

    household_features = household_education.merge(
        household_size,
        on="hhid",
        how="left"
    )

    print(household_features.head())

    print(household_features.isnull().sum())


    return household_features

def load_household_gender_data():

    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect1_harvestw5.csv"

    household = pd.read_csv(path)

    household = household[[
        "hhid",
        "s1q2",
        "s1q3"
    ]]

    household = household[
        household["s1q3"].astype(str).str.contains("HEAD", case=False, na=False)
    ]

    household["head_gender"] = household["s1q2"].map({
        "1. MALE": 1,
        "2. FEMALE": 0,
        1: 1,
        2: 0
    })

    household = household[[
        "hhid",
        "head_gender"
    ]]

    print(household.head())
    print(household["head_gender"].value_counts())
    print(household.isnull().sum())

    return household

def load_shock_data():

    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect12_harvestw5.csv"

    df = pd.read_csv(path)


    print(df[["hhid", "shock_cd", "s12q1"]].head())

    # Keep only households affected by shock
    df = df[
        (df["s12q1"] == "1. YES") |
        (df["s12q1"] == 1)
    ]

    # Count number of shocks per household
    shock_counts = df.groupby("hhid").size().reset_index(name="num_shocks")

    # Create shock level
    shock_counts["shock_level"] = 0

    shock_counts.loc[
        shock_counts["num_shocks"] == 1,
        "shock_level"
    ] = 1

    shock_counts.loc[
        shock_counts["num_shocks"] >= 2,
        "shock_level"
    ] = 2

    shock_counts = shock_counts[[
        "hhid",
        "shock_level"
    ]]

    print(shock_counts.head())

    print(shock_counts["shock_level"].value_counts())

    print(shock_counts.isnull().sum())
    print(shock_counts.shape)

    return shock_counts

def load_assistance_data():

    print("ASSISTANCE FUNCTION IS RUNNING")

    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect11a_harvestw5.csv"


    df = pd.read_csv(path)

    print(df.shape)
    print(df.head())

    df = df[[
            "hhid",
            "s11q1a"
    ]]

    df["received_assistance"] = df["s11q1a"].map({
            "1. YES": 1,
            "2. NO": 0,
            1: 1,
            2: 0
    })

    df = df[[
            "hhid",
            "received_assistance"
    ]]

    print(df.head())
    print(df["received_assistance"].value_counts())
    print(df.isnull().sum())

    return df

def load_fertilizer_data():

    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\secta11c2_harvestw5.csv"

    df = pd.read_csv(path)

    print(df.shape)

    df = df[[
        "hhid",
        "s11c2q5",
        "s11c2q11"
    ]]

    # Inorganic fertilizer
    inorganic = (
        (df["s11c2q5"] == "1. YES") |
        (df["s11c2q5"] == 1)
    )

    # Organic fertilizer
    organic = (
        (df["s11c2q11"] == "1. YES") |
        (df["s11c2q11"] == 1)
    )

    # Create fertilizer feature
    df["used_fertilizer"] = 0

    df.loc[inorganic | organic, "used_fertilizer"] = 1

    # Household-level aggregation
    fertilizer = df.groupby("hhid")["used_fertilizer"].max().reset_index()

    print(fertilizer.head())

    print(fertilizer["used_fertilizer"].value_counts())

    print(fertilizer.isnull().sum())

    return fertilizer


def load_plot_size_data():

    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Planting Wave 5\Agriculture\sect11a1_plantingw5.csv"

    df = pd.read_csv(path)



    print(df.shape)

    df = df[[
        "hhid",
        "s11aq3_number",
        "s11aq3_unit"
    ]]

    print(df.head())

    # Convert units to approximate square meters

    df["land_size"] = 0.0

    df.loc[df["s11aq3_unit"] == "7. SQUARE METERS", "land_size"] = df["s11aq3_number"]

    df.loc[df["s11aq3_unit"] == "6. HECTARES", "land_size"] = (
            df["s11aq3_number"] * 10000
    )

    df.loc[df["s11aq3_unit"] == "5. ACRES", "land_size"] = (
            df["s11aq3_number"] * 4046.86
    )

    # Aggregate to household level

    plot_size = df.groupby("hhid")["land_size"].sum().reset_index()

    print(plot_size.head())

    print(plot_size["land_size"].describe())

    print(plot_size.isnull().sum())

    plot_size["land_size"] = plot_size["land_size"].clip(
        upper=plot_size["land_size"].quantile(0.99)
    )

    print(plot_size["land_size"].describe())

    return plot_size


def load_credit_access_data():

    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\secta12_harvestw5.csv"

    df = pd.read_csv(path)

    print([col for col in df.columns if "s12q20" in col])

    df = df[[
        "hhid",
        "s12q20__3"
    ]]
    print(df["s12q20__3"].value_counts(dropna=False))

    df["received_credit"] = df["s12q20__3"].fillna(0)

    df["received_credit"] = df["received_credit"].astype(int)


    df = df[[
        "hhid",
        "received_credit"
    ]]

    df = df.groupby("hhid")["received_credit"].max().reset_index()

    print(df.head())

    print(df["received_credit"].value_counts())

    print(df.isnull().sum())

    return df

def load_zone_data():

    path = r"C:\Users\ALALI\Documents\STUDY\MY DOCUMENTS\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect1_harvestw5.csv"

    df = pd.read_csv(path)

    df = df[[
        "hhid",
        "zone"
    ]]

    print(df.isnull().sum())

    df["zone"] = df["zone"].astype("category").cat.codes

    df = df.drop_duplicates(subset="hhid")

    return df


def check_roster_columns():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect1_harvestw5.csv"

    df = pd.read_csv(path)

    print(df.shape)
    print(df.columns)
    print(df.head())

    return df


check_roster_columns()

def load_dependency_ratio_data():
    path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Household\sect1_harvestw5.csv"

    df = pd.read_csv(path)

    df = df[[
        "hhid",
        "s1q6"
    ]]

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

    dependency = dependency[[
        "hhid",
        "dependency_ratio"
    ]]

    print(dependency.head())
    print(dependency["dependency_ratio"].describe())
    print(dependency.isnull().sum())

    return dependency

load_dependency_ratio_data()

def load_asset_data():
    asset_path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Planting Wave 5\Household\sect10_plantingw5.csv"

    assets = pd.read_csv(asset_path)

    owned_assets = assets[assets["s10q1a"] == "1. YES"].copy()

    owned_assets["s10q2"] = owned_assets["s10q2"].fillna(1)

    asset_features = (
        owned_assets
        .groupby("hhid")["s10q2"]
        .sum()
        .reset_index()
    )

    asset_features.rename(columns={"s10q2": "asset_score"}, inplace=True)

    asset_cap = asset_features["asset_score"].quantile(0.99)

    asset_features["asset_score"] = asset_features["asset_score"].clip(upper=asset_cap)

    return asset_features

def load_transport_cost_data():

    transport_path = r"C:\Users\ALALI\Desktop\AgroReach\DATASET\NGA_2023_GHSP-W5_v01_M_CSV\Post Harvest Wave 5\Agriculture\secta11c3q12_harvestw5.csv"

    transport = pd.read_csv(transport_path)

    transport_features = transport[["hhid", "s11c3q12"]].copy()

    transport_features.rename(
        columns={"s11c3q12": "transport_cost"},
        inplace=True
    )

    transport_features["transport_cost"] = (
        transport_features["transport_cost"]
        .astype(str)
        .str.extract(r"(\d+)")
    )

    transport_features["transport_cost"] = pd.to_numeric(
        transport_features["transport_cost"],
        errors="coerce"
    )

    transport_features["transport_cost"] = transport_features["transport_cost"].fillna(0)

    return transport_features