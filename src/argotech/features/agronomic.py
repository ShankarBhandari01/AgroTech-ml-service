"""The single feature builder, shared by training and serving.

There is exactly one implementation of every feature, called from both paths with the same daily
weather block shape (`data.meteo.daily_frame`) and the same satellite index dict. That is the
structural fix for the train/serve skew described in docs/model-design.md: a feature cannot mean two
different things if it is only defined once.

All computation delegates to `argotech.domain`, which is pure and separately tested.
"""

from __future__ import annotations

from argotech.domain import agronomy, indices

# Length of the look-back window. 92 days is the Open-Meteo forecast endpoint's maximum look-back,
# so the serving path can cover the same window the training path uses.
WINDOW_DAYS = 90

# Column order is the model contract. Appending is safe; reordering or renaming requires a retrain.
FEATURE_COLUMNS = [
    # --- thermal ---
    "gdd_90", "gdd_since_onset", "tmax_mean_30", "tmin_mean_30", "diurnal_range_30",
    "heat_stress_days",
    # --- water ---
    "rain_30", "rain_90", "et0_90", "water_satisfaction_30", "water_deficit_30",
    "dry_spell_30", "dry_spell_90", "rain_anomaly_30",
    # --- atmosphere ---
    "rh_mean_30", "radiation_90",
    # --- phenology ---
    "days_since_onset", "stage_kc",
    # --- canopy state ---
    "ndvi", "ndmi", "evi", "vci", "ndvi_z_peer",
    # --- site ---
    "latitude", "longitude", "elevation",
]


def _slice(daily: dict, days: int) -> dict:
    return {k: v[-days:] for k, v in daily.items() if k != "time"}


def build(daily: dict, sat: dict, site: dict, crop: str = agronomy.DEFAULT_CROP) -> dict:
    """One feature row.

    `daily`   — output of `data.meteo.daily_frame`, ending on the prediction date.
    `sat`     — {ndvi, ndmi, evi, vci, ndvi_z_peer} for the prediction date.
    `site`    — {latitude, longitude, elevation, clim_rain_30}, the last being the site's own
                climatological 30-day rainfall for this time of year, used for a *real* anomaly.
    """
    w30, w90 = _slice(daily, 30), _slice(daily, WINDOW_DAYS)

    tmax90, tmin90 = w90["temperature_2m_max"], w90["temperature_2m_min"]
    rain90, et090 = w90["precipitation_sum"], w90["et0_fao_evapotranspiration"]
    tmax30, tmin30 = w30["temperature_2m_max"], w30["temperature_2m_min"]
    rain30, et030 = w30["precipitation_sum"], w30["et0_fao_evapotranspiration"]

    gdd_daily = [agronomy.growing_degree_days(lo, hi) for lo, hi in zip(tmin90, tmax90)]

    # Phenology, anchored on the rainy-season onset within the window rather than assumed.
    onset = agronomy.season_onset_index(rain90)
    if onset is None:
        days_since_onset, gdd_since_onset = 0, 0.0
    else:
        days_since_onset = len(rain90) - onset
        gdd_since_onset = sum(gdd_daily[onset:])
    stage = agronomy.phenology_stage(gdd_since_onset, crop)

    wb30 = agronomy.water_balance(rain30, et030, stage)
    wb90 = agronomy.water_balance(rain90, et090, stage)

    return {
        "gdd_90": round(sum(gdd_daily), 1),
        "gdd_since_onset": round(gdd_since_onset, 1),
        "tmax_mean_30": round(sum(tmax30) / len(tmax30), 2),
        "tmin_mean_30": round(sum(tmin30) / len(tmin30), 2),
        "diurnal_range_30": round((sum(tmax30) - sum(tmin30)) / len(tmax30), 2),
        "heat_stress_days": agronomy.heat_stress_days(tmax30, stage),

        "rain_30": round(sum(rain30), 1),
        "rain_90": round(sum(rain90), 1),
        "et0_90": round(sum(et090), 1),
        "water_satisfaction_30": wb30.satisfaction,
        "water_deficit_30": wb30.deficit_mm,
        "dry_spell_30": wb30.longest_dry_spell_days,
        "dry_spell_90": wb90.longest_dry_spell_days,
        # A real anomaly: observed minus this site's own climatological normal for the same window.
        "rain_anomaly_30": round(sum(rain30) - site.get("clim_rain_30", sum(rain30)), 1),

        "rh_mean_30": round(sum(w30["relative_humidity_2m_mean"]) / len(w30["relative_humidity_2m_mean"]), 1),
        "radiation_90": round(sum(w90["shortwave_radiation_sum"]), 1),

        "days_since_onset": days_since_onset,
        "stage_kc": agronomy.KC_BY_STAGE.get(stage, 1.0),

        "ndvi": round(sat["ndvi"], 4),
        "ndmi": round(sat["ndmi"], 4),
        "evi": round(sat["evi"], 4),
        "vci": round(sat["vci"], 2),
        "ndvi_z_peer": round(sat["ndvi_z_peer"], 3),

        "latitude": site["latitude"],
        "longitude": site["longitude"],
        "elevation": site.get("elevation", 0.0),
    }


def satellite_block(current: dict, past_ndvi: list[float], peer_ndvi: list[float]) -> dict:
    """Canopy-state features. `past_ndvi` is this site's observations strictly *before* now (so VCI
    carries no future information); `peer_ndvi` is the concurrent cohort, excluding this site."""
    ndvi = current["ndvi"]
    history = past_ndvi or [ndvi]
    peer = peer_ndvi or [ndvi]
    mean = sum(peer) / len(peer)
    var = sum((p - mean) ** 2 for p in peer) / len(peer)
    return {
        "ndvi": ndvi,
        "ndmi": current["ndwi"],   # the repo's "ndwi" is the B08/B11 formula, i.e. NDMI
        "evi": current["evi"],
        "vci": indices.vci(ndvi, min(history), max(history)),
        "ndvi_z_peer": indices.anomaly_z(ndvi, mean, var ** 0.5),
    }
