"""E07 step 2 — build `exposure` for real, from GHS-Panel Wave 5.

`domain.risk.assess_exposure(area_ha, expected_yield_t_ha, price_per_t)` has never had data behind
it. Notebook 05 could only sweep illustrative ranges. This builds all three inputs from the survey.

IDENTIFICATION. The CSV export carries no variable labels, so nothing here trusts a column name.
Each mapping below is asserted by evidence and carries a confidence level:

  area_ha   <- sect11a1.SR_hect                                             CONFIRMED
      SR_meter / SR_hect == 10000.0 for 100.00% of 9,231 plots -- the exact ha->m2 constant, which
      no other pair of quantities would satisfy. Magnitudes are agronomically sane (p25 0.046,
      median 0.186, p95 2.00 ha) and aggregate to a median farm of 0.59 ha over 4,002 households.
      A GPS-measured subsample (s11mq3, n=5,137) runs at 0.88x the self-report -- farmers over-state
      plot size by ~14% at the median, spearman +0.549. That bias propagates into every yield below.

  harvest_kg <- secta3i.sa3iq9a * secta3i.sa3iq9_conv                       CONFIRMED
      Units are local and labelled ('150. BASIN', '130. SACK/BAG', '211. TUBER'), which is exactly
      why a conversion factor column exists. Median 252 kg per crop-plot.

  price_per_t <- secta3i.sa3iq10 / harvest_kg                               CONFIRMED
      Yields 400 NGN/kg = 400,000 NGN/tonne ~= 267 USD/t at 1,500 NGN/USD. The repository's own
      FARMGATE_PRICE_USD_PER_T constants span 120-420 USD/t, so an independently derived 267 lands
      inside a range this codebase set before the survey was ever opened.

  yield_t_ha <- harvest_kg / 1000 / area_ha                                 CROP-DEPENDENT
      Validated against FAO Nigeria averages on SOLE-CROPPED plots only (41.3% of plots carry more
      than one crop, and dividing whole-plot area by each crop understates every one of them):
        rice 1.75 vs ~2.0 and sorghum 0.84 vs ~1.0   -- agree
        maize 0.80 vs ~1.8                           -- 2x low, consistent with the known gap
                                                        between LSMS self-reports and official stats
        cassava 2.50 vs ~12                          -- 5x low; cassava is harvested PIECEMEAL over
                                                        months, so a post-harvest visit captures
                                                        harvest-to-date, not the season
      So yield is usable for rice and sorghum, weak for maize, and NOT usable for cassava. The
      builder therefore records a per-crop trust flag rather than emitting one undifferentiated
      number.

NOT identified, and deliberately not guessed: the seven COPING_FACTORS (irrigation, extension,
credit, fertilizer, asset score, market access) sit behind bare question codes. `crop_diversity` is
the exception -- it is structural, counted from `cropcode`, and needs no label at all.

Run: venv/bin/python3 experiments/E07-lsms/exposure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/lsms/NGA_2023_GHSP-W5_v01_M_CSV")
OUT = Path("experiments/E07-lsms")
NGN_PER_USD = 1500.0

# FAO-ish Nigeria national averages, used only to flag trust, never to correct a value.
FAO_T_HA = {"1080. MAIZE": 1.8, "1110. RICE": 2.0, "1070. GUINEA CORN (SORGHUM)": 1.0,
            "1020. CASSAVA": 12.0, "1100. MILLET/MAIWA": 1.0, "1010. BEANS/COWPEA": 0.8}
TRUSTED = {"1110. RICE", "1070. GUINEA CORN (SORGHUM)", "1100. MILLET/MAIWA"}


def main() -> None:
    h = pd.read_csv(ROOT/"Post Harvest Wave 5/Agriculture/secta3i_harvestw5.csv", low_memory=False)
    a = pd.read_csv(ROOT/"Post Planting Wave 5/Agriculture/sect11a1_plantingw5.csv", low_memory=False)

    # --- guard the join before making it: a duplicated plot key would multiply harvest rows ---
    assert not a.duplicated(["hhid", "plotid"]).any(), "plot roster is not unique on (hhid, plotid)"

    h["harvest_kg"] = h["sa3iq9a"] * h["sa3iq9_conv"]
    j = h.merge(a[["hhid", "plotid", "SR_hect", "s11mq3"]], on=["hhid", "plotid"],
                how="inner", validate="many_to_one")
    before = len(j)
    j = j[(j["SR_hect"] > 0) & (j["harvest_kg"] > 0)].copy()
    print(f"crop-plot rows: {before} joined, {len(j)} with positive area and harvest")

    j["n_crops_on_plot"] = j.groupby(["hhid", "plotid"])["cropcode"].transform("nunique")
    j["sole_cropped"] = j["n_crops_on_plot"] == 1
    j["yield_t_ha"] = (j["harvest_kg"] / 1000.0) / j["SR_hect"]
    j["price_ngn_per_t"] = j["sa3iq10"] / (j["harvest_kg"] / 1000.0)
    j["price_usd_per_t"] = j["price_ngn_per_t"] / NGN_PER_USD
    j["yield_trusted"] = j["cropcode"].isin(TRUSTED)
    # GPS area where measured, so the self-report bias can be quantified downstream rather than
    # silently inherited. Never substituted -- both are carried.
    j["area_ha_gps"] = np.where(j["s11mq3"] > 0, j["s11mq3"] / 10000.0, np.nan)

    # --- exposure, the quantity assess_exposure() wants ---
    j["value_at_risk_usd"] = j["SR_hect"] * j["yield_t_ha"] * j["price_usd_per_t"]

    # crop_diversity: structural, no codebook needed
    div = j.groupby("hhid")["cropcode"].nunique().rename("crop_diversity")

    hh = (j.groupby("hhid")
            .agg(area_ha=("SR_hect", "sum"),
                 n_plots=("plotid", "nunique"),
                 n_crop_rows=("cropcode", "size"),
                 value_at_risk_usd=("value_at_risk_usd", "sum"),
                 median_yield_t_ha=("yield_t_ha", "median"),
                 median_price_usd_t=("price_usd_per_t", "median"),
                 share_sole_cropped=("sole_cropped", "mean"),
                 share_yield_trusted=("yield_trusted", "mean"))
            .join(div))

    print(f"\nhousehold-level exposure: {len(hh)} households")
    q = hh["value_at_risk_usd"].quantile([.05, .25, .5, .75, .95])
    print("  value_at_risk (USD):", {f"p{int(k*100)}": round(v, 1) for k, v in q.items()})
    print(f"  median farm {hh.area_ha.median():.2f} ha, median crop diversity {hh.crop_diversity.median():.0f}")
    print(f"  households whose yields are ALL from trusted crops: "
          f"{(hh.share_yield_trusted == 1).sum()} ({(hh.share_yield_trusted == 1).mean():.1%})")

    print("\nper-crop summary (sole-cropped plots only, the honest denominator):")
    s = j[j.sole_cropped]
    for crop, fao in sorted(FAO_T_HA.items(), key=lambda kv: -kv[1]):
        sub = s[s.cropcode == crop]
        if len(sub) < 30:
            continue
        med = sub.yield_t_ha.median()
        print(f"  {crop:32s} n={len(sub):>5} yield={med:>6.2f} t/ha  FAO~{fao:<5} "
              f"ratio={med/fao:>5.2f}  {'trusted' if crop in TRUSTED else 'NOT trusted'}")

    j.to_parquet(OUT/"exposure_crop_plot.parquet")
    hh.to_parquet(OUT/"exposure_household.parquet")
    summary = {
        "n_crop_plot_rows": int(len(j)), "n_households": int(len(hh)),
        "n_plots": int(j.groupby(["hhid", "plotid"]).ngroups),
        "ngn_per_usd_assumed": NGN_PER_USD,
        "median_area_ha": float(hh.area_ha.median()),
        "median_value_at_risk_usd": float(hh.value_at_risk_usd.median()),
        "median_price_usd_per_t": float(j.price_usd_per_t.median()),
        "share_plots_intercropped": float((j.n_crops_on_plot > 1).mean()),
        "gps_to_selfreport_area_ratio": float((j.area_ha_gps / j.SR_hect).median()),
        "trusted_crops": sorted(TRUSTED),
    }
    (OUT/"exposure_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {OUT}/exposure_household.parquet and exposure_summary.json")


if __name__ == "__main__":
    main()
