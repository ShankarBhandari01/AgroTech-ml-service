"""E10 step 1 — assemble a crop-loss dataset from GHS-Panel W5, with the leakage screen up front.

TARGET. `sa3iq6` = "was the area harvested of [CROP] on [PLOT] less than the area planted?", with
the censoring category removed: the largest single "cause" is *Have Not Completed Harvest* (695 of
2,647, 26.3%), which is interview timing, not loss. Leaving it in teaches a model to predict when
the enumerator visited. Grain: plot x crop.

This is the first supervised agronomic outcome available anywhere in this project -- the product's
own `field_outcomes` table is still empty (`data/store.py:label_join`).

LEAKAGE SCREEN, applied by RULE before any statistic is computed:

  1. Nothing derived from the harvest may predict a harvest shortfall. `yield_t_ha`, `harvest_kg`,
     `value_at_risk_usd`, `price_usd_per_t`, `sa3iq9a`, `sa3iq10` are all computed FROM the harvest
     whose shortfall is the label. Including them is circular.
  2. Nothing from the shock module (`sect10`). Those questions are asked of the same household in
     the same interview about the same season -- "did you face difficulty due to drought" is a
     restatement of the outcome, not a predictor of it. They are used in the EDA to VALIDATE the
     label (notebook 08 section 10), never to fit it.
  3. No protected attributes. `head_gender` and `household_max_education` are rejected at runtime by
     `risk.assess_vulnerability`; the same rule applies here.
  4. `sa3iq7_*` / `sa3iq13_*` are the attributed CAUSE of the loss -- recorded only when a loss
     occurred. Pure target leakage.

Run: venv/bin/python3 experiments/E10-croploss/build_dataset.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = "data/lsms/NGA_2023_GHSP-W5_v01_M_Stata/"
OUT = Path("experiments/E10-croploss")

BANNED = {
    "harvest-derived (circular)": ["sa3iq9a", "sa3iq9b", "sa3iq9_conv", "sa3iq10", "sa3iq15a",
                                   "yield_t_ha", "harvest_kg", "value_at_risk_usd", "price_usd_per_t"],
    "same-interview restatement": ["s10q1a", "s10q2a", "s10q3a", "s10q4a", "s10q5a", "s10q6a",
                                   "s10q7a", "s10q9a"],
    "attributed cause of the label": ["sa3iq7_1", "sa3iq7_2", "sa3iq12", "sa3iq13_1", "sa3iq13_2"],
    "protected": ["head_gender", "household_max_education"],
}


def dta(name: str, cat: bool = False) -> pd.DataFrame:
    return pd.read_stata(glob.glob(R + "**/" + name, recursive=True)[0], convert_categoricals=cat)


def main() -> None:
    h = dta("secta3i_harvestw5.dta", cat=True)
    loss_raw = h.sa3iq6.astype(str).str.startswith("1")
    censored = h.sa3iq7_1.astype(str).str.contains("Not Completed", case=False, na=False)
    df = h[["hhid", "plotid", "cropcode", "ea", "state", "zone"]].copy()
    df["crop_loss"] = (loss_raw & ~censored).astype(int)
    # A row that is censored carries no usable label: we do not know whether the remaining area
    # would have been harvested. Drop rather than call it "no loss".
    df = df[~(loss_raw & censored)]
    print(f"label rows {len(df)}  loss rate {df.crop_loss.mean():.1%}  "
          f"(dropped {int((loss_raw & censored).sum())} censored)")

    # --- plot-level predictors, all pre-harvest -----------------------------------------------
    a1 = dta("sect11a1_plantingw5.dta")[["hhid", "plotid", "SR_hect", "s11mq3"]]
    a1 = a1.rename(columns={"s11mq3": "gps_area_m2"})
    b1 = dta("sect11b1_plantingw5.dta")[["hhid", "plotid", "s11b1q56", "s11b1q11", "s11b1q43"]]
    b1 = b1.rename(columns={"s11b1q56": "irrigated_raw", "s11b1q11": "rent_paid",
                            "s11b1q43": "plot_value_ngn"})
    c2 = dta("secta11c2_harvestw5.dta")[["hhid", "plotid", "s11c2q5", "s11c2q14"]]
    c2 = c2.rename(columns={"s11c2q5": "fert_raw", "s11c2q14": "traction_raw"})
    # planting month: the onset-timing signal, and "late onset of rains" is the single largest
    # attributed cause of loss (562 rows) -- see notebooks/08-lsms-eda.ipynb section 11.
    f = dta("sect11f_plantingw5.dta")
    plant = (f[["hhid", "plotid", "s11fq11_month"]].dropna()
             .groupby(["hhid", "plotid"]).s11fq11_month.median().rename("plant_month").reset_index()
             if "s11fq11_month" in f else None)

    for frame, keys in ((a1, ["hhid", "plotid"]), (b1, ["hhid", "plotid"]), (c2, ["hhid", "plotid"])):
        frame.drop_duplicates(keys, inplace=True)
    df = df.merge(a1, on=["hhid", "plotid"], how="left", validate="many_to_one")
    df = df.merge(b1, on=["hhid", "plotid"], how="left", validate="many_to_one")
    df = df.merge(c2, on=["hhid", "plotid"], how="left", validate="many_to_one")
    if plant is not None:
        df = df.merge(plant, on=["hhid", "plotid"], how="left", validate="many_to_one")

    # --- household-level coping capacity (already built in E07) --------------------------------
    vu = pd.read_parquet("experiments/E07-lsms/vulnerability_household.parquet")
    df = df.merge(vu[["has_extension_access", "asset_score", "market_access_score", "crop_diversity"]],
                  left_on="hhid", right_index=True, how="left")

    # --- decode 1=YES / 2=NO, the trap that inverts every boolean in this survey ---------------
    for raw, clean in (("irrigated_raw", "irrigated"), ("fert_raw", "used_fertilizer"),
                       ("traction_raw", "animal_traction")):
        df[clean] = df[raw].eq(1).astype(float)
        df.drop(columns=[raw], inplace=True)

    df["n_crops_on_plot"] = df.groupby(["hhid", "plotid"]).cropcode.transform("nunique")
    df["crop"] = df.cropcode.astype(str).str.split(".").str[-1].str.strip()

    banned_present = {k: [c for c in v if c in df.columns] for k, v in BANNED.items()}
    assert not any(banned_present.values()), f"leakage survived the screen: {banned_present}"

    df.to_parquet(OUT / "dataset.parquet")
    (OUT / "leakage_screen.json").write_text(json.dumps(
        {"banned_by_rule": BANNED, "n_rows": int(len(df)), "loss_rate": float(df.crop_loss.mean()),
         "n_ea": int(df.ea.nunique()), "n_households": int(df.hhid.nunique())}, indent=1))
    print(f"rows {len(df)}  households {df.hhid.nunique()}  EAs {df.ea.nunique()}  crops {df.crop.nunique()}")
    print("features:", [c for c in df.columns if c not in
                        ("hhid", "plotid", "cropcode", "ea", "state", "zone", "crop_loss")])
    print(f"\nwrote {OUT}/dataset.parquet")


if __name__ == "__main__":
    main()
