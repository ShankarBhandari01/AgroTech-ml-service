"""The single feature builder, shared by training and serving.

There is exactly one implementation of every feature, called from both paths with the same daily
weather block shape (`data.meteo.daily_frame`) and the same satellite index dict. That is the
structural fix for the train/serve skew described in docs/model-design.md: a feature cannot mean two
different things if it is only defined once.

All computation delegates to `argotech.domain`, which is pure and separately tested.
"""

from __future__ import annotations

import math
from datetime import date

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
    # --- canopy structure, radar (Sentinel-1) ---
    # Present for every field, cloud or not, which is the point: optical gaps cluster in the rainy
    # season. Missing (NaN) where a site has no S1 coverage within tolerance of the optical date.
    "rvi", "vh_vv_ratio", "rvi_z_peer",
]


# ---------------------------------------------------------------------------------------------
# What the *model* consumes, as opposed to what gets computed and stored.
#
# These are deliberately separate. FEATURE_COLUMNS is the `field_features` schema — changing it
# migrates a table. MODEL_FEATURES is a modelling decision, and a retrain is the only thing it
# costs. Keeping them fused would mean every experiment on the feature set is a schema change.
# ---------------------------------------------------------------------------------------------

# Permutation importance on a held-out cluster scores these at or below zero, and most are algebraic
# restatements of columns we keep: vci and rain_anomaly_30 are normalisations of ndvi and rain_30,
# water_deficit_30 is the complement of water_satisfaction_30, gdd_since_onset tracks gdd_90.
# `longitude` is pure cluster identity, which is exactly what we are trying to normalise away.
UNINFORMATIVE = frozenset({
    "vci", "water_deficit_30", "rain_anomaly_30", "longitude", "gdd_since_onset", "tmin_mean_30",
})

# Features that carry a strong regional signature and so get a within-cluster twin. The cross-region
# generalisation literature (leave-one-country-out on sub-Saharan maize, arXiv 2605.08113) finds that
# label distribution shift is the binding constraint; our label is already peer-standardised within
# cluster x date, but these inputs still hand the model raw cluster identity.
CLUSTER_RELATIVE = [
    "gdd_90", "tmax_mean_30", "diurnal_range_30", "heat_stress_days",
    "rain_30", "rain_90", "et0_90", "water_satisfaction_30", "dry_spell_30", "dry_spell_90",
    "rh_mean_30", "radiation_90", "ndvi", "ndmi", "evi",
    # Backscatter is strongly land-cover and terrain dependent, so its regional signature is at
    # least as strong as the optical bands' — it needs the twin more, not less.
    "rvi", "vh_vv_ratio",
]

CZ_SUFFIX = "_cz"

# Sentinel-1 derived columns, named so an ablation can drop exactly the radar contribution and
# nothing else. The comparison that matters is radar-on vs radar-off on the *same* parquet: the
# dataset builder keys its window off `date.today()`, so two builds on different days are not a
# controlled comparison.
RADAR_FEATURES = frozenset({"rvi", "vh_vv_ratio", "rvi_z_peer",
                            "rvi" + CZ_SUFFIX, "vh_vv_ratio" + CZ_SUFFIX})

MODEL_FEATURES = (
    [c for c in FEATURE_COLUMNS if c not in UNINFORMATIVE]
    + [c + CZ_SUFFIX for c in CLUSTER_RELATIVE]
)


def add_cluster_relative(df, group: str = "cluster"):
    """Append a within-group z-score twin for each column in `CLUSTER_RELATIVE`.

    Standardisation uses only feature values — never the label — so a held-out cluster computing its
    own mean and standard deviation is not leakage. It is the point: the transform is what strips
    cluster identity out of the inputs.

    `mu_c` and `sigma_c` are pooled over the whole frame per cluster — one pair per (cluster,
    column), not per date. Serving must reuse *these* constants rather than recomputing from its own
    rows: the model was fitted on twins measured against this reference distribution, and z-scores
    against any other one are a different feature wearing the same name. `cluster_stats` snapshots
    them into the artifact and `cluster_relative_row` replays them one row at a time.

    For feature x and cluster c:

        mu_c    = (1 / n_c) * SUM_{i in c} x_i
        sigma_c = sqrt( (1 / (n_c - 1)) * SUM_{i in c} (x_i - mu_c)^2 )     (pandas std, ddof=1)
        x_cz    = (x_i - mu_c) / sigma_c            if sigma_c > 1e-9 and x_i is observed
                = 0                                 if the column is constant within the cluster
                = NaN                               if x_i itself is missing

    mu_c and sigma_c skip missing values (pandas default), so a partially-observed column still
    standardises against the fields that *were* observed.

    The three-way split matters. A constant column carries no information and 0.0 says so honestly.
    But a *missing* value is not a value at the cluster mean — collapsing it to 0.0 would tell the
    model a cloud-obscured field is an average field, which is exactly the fabrication `radar_block`
    and `satellite_block` refuse to make. Radar and optical columns both carry real NaN, so this is
    load-bearing rather than defensive.
    """
    out = df.copy()
    grouped = out.groupby(group)
    for col in CLUSTER_RELATIVE:
        mean = grouped[col].transform("mean")
        sd = grouped[col].transform("std")
        twin = ((out[col] - mean) / sd).where(sd > 1e-9, 0.0)
        # Restore missingness that the constant-column fallback would otherwise have filled in.
        out[col + CZ_SUFFIX] = twin.where(out[col].notna())
    return out


# The raw columns whose peer anomaly is standardised. Their `_z_peer` outputs are model features;
# `ndvi_z_peer` is also the entire input to the persistence hazard path.
PEER_RELATIVE = ["ndvi", "rvi"]


def peer_bucket(cluster: str | None, sensing_date: str) -> str | None:
    """The reference bucket for one observation: `"<cluster>|<MM>"`, or None without a cluster.

    Calendar month, not date: the cohort a field is compared against has to be knowable at serving
    time, and a specific date's cohort is not. Month is the coarsest key that still separates the
    growing season from the dry season, which is the variation that would otherwise dominate.
    """
    return f"{cluster}|{sensing_date[5:7]}" if cluster else None


def peer_stats(df, group: str = "cluster") -> dict:
    """`{"<cluster>|<MM>": {column: [mu, sigma]}}` — the reference both paths standardise against.

    Snapshotted into the artifact so serving reproduces training's arithmetic from constants rather
    than from whatever sample it happens to hold. Plain lists rather than a DataFrame: this crosses
    a joblib boundary and then a `predict` hot path, and must not drag pandas into either.

    Callers today: `lab.panel.py` and the incumbent `training.train.py` both call this over the
    whole frame — `panel.py` to bake in the `leaky` control column `lab.run` compares against, and
    `train.py` to snapshot the reference the shipped artifact serves from. `serving.pipeline` then
    reads that snapshot at inference, which is correct there: a live request is one row with no
    future to leak from. The lab's honest evaluation does not call this function over the whole
    frame at all — `argotech.lab.peers.fit_peer_stats` mirrors its output shape but is fit on
    training rows only, per fold, so a held-out cluster is never standardised against itself.
    """
    keys = df[group].astype(str) + "|" + df["obs_date"].str.slice(5, 7)
    grouped = df.assign(_bucket=keys).groupby("_bucket")
    mean, sd = grouped[PEER_RELATIVE].mean(), grouped[PEER_RELATIVE].std()
    return {str(b): {col: [float(mean.at[b, col]), float(sd.at[b, col])] for col in PEER_RELATIVE}
            for b in mean.index}


def peer_reference(stats: dict | None, bucket: str | None, column: str) -> list[float] | None:
    """One `[mu, sigma]`, or None when the bucket was never measured.

    None rather than a default: `[0.0, 1.0]` would assert that an unmeasured region is exactly
    average, which is the fabrication `cluster_relative_row` refuses for the same reason.
    """
    if stats is None or bucket is None:
        return None
    entry = (stats.get(bucket) or {}).get(column)
    return entry if entry and math.isfinite(entry[0]) else None


def cluster_stats(df, group: str = "cluster") -> dict:
    """`{cluster: {column: [mu, sigma]}}` — the reference distribution `add_cluster_relative` used.

    Snapshotted into the model artifact so the serving path can standardise a single row against
    the same constants. Plain lists rather than a DataFrame: this crosses a joblib boundary and
    then a `predict` hot path, and it must not drag pandas into either.
    """
    grouped = df.groupby(group)
    mean, sd = grouped[CLUSTER_RELATIVE].mean(), grouped[CLUSTER_RELATIVE].std()
    return {str(c): {col: [float(mean.at[c, col]), float(sd.at[c, col])] for col in CLUSTER_RELATIVE}
            for c in mean.index}


def cluster_relative_row(row: dict, stats: dict | None) -> dict:
    """The `_cz` twins for one row, reproducing `add_cluster_relative`'s three-way split exactly.

    `stats` is one cluster's `{column: [mu, sigma]}`, or None for a field outside every known
    cluster. An unknown cluster yields NaN rather than 0.0 for every twin: 0.0 asserts "average for
    its region", and inventing that for a region we have never measured is the same fabrication the
    docstring above refuses to make for a cloud-obscured field. HistGradientBoosting splits on NaN
    natively, so the model degrades to the raw features instead of being lied to.
    """
    out = {}
    for col in CLUSTER_RELATIVE:
        x, mu_sigma = row.get(col), (stats or {}).get(col)
        if x is None or not math.isfinite(x) or mu_sigma is None or not math.isfinite(mu_sigma[0]):
            out[col + CZ_SUFFIX] = float("nan")
            continue
        mu, sigma = mu_sigma
        # sigma > 1e-9 mirrors the vectorised `.where(sd > 1e-9, 0.0)`; a NaN sigma (a cluster with
        # fewer than two observations of this column) is not > 1e-9 and so lands on 0.0 the same way.
        out[col + CZ_SUFFIX] = (x - mu) / sigma if math.isfinite(sigma) and sigma > 1e-9 else 0.0
    return out


def assign_cluster(latitude: float, longitude: float, bounds: dict | None) -> str | None:
    """Which training cluster a field falls in, or None if it falls in none of them.

    Bounding boxes rather than a nearest-centroid match, because the clusters are not a partition of
    the continent — they are six sampled zones. Snapping a farm 800 km away to the nearest one would
    standardise it against a region it has nothing in common with.
    """
    for name, box in (bounds or {}).items():
        (lat0, lat1), (lon0, lon1) = box["lat"], box["lon"]
        if lat0 <= latitude <= lat1 and lon0 <= longitude <= lon1:
            return name
    return None


def _slice(daily: dict, days: int) -> dict:
    return {k: v[-days:] for k, v in daily.items() if k != "time"}


def _round_or_nan(value: float, digits: int) -> float:
    """`round` propagates NaN fine, but not None — and an absent radar block gives either."""
    return float("nan") if value is None or value != value else round(value, digits)


def finite(value) -> bool:
    """True for a real, usable measurement.

    Cached upstream responses predate the current filtering and can still contain JSON `NaN`, and a
    NaN peer silently poisons a whole cohort's mean and standard deviation — every z-score computed
    against it becomes NaN. On Python 3.12 `statistics.pstdev` additionally *raises* on such a list
    ("'float' object has no attribute 'numerator'"), so the same defect is a crash on one interpreter
    and silent corruption on another. Filtering at cohort-construction time fixes both.
    """
    return isinstance(value, (int, float)) and value == value and value not in (float("inf"), float("-inf"))


def nearest_sar(sar: list[dict], target: str, max_gap_days: int = 20) -> dict | None:
    """The radar observation closest in time to an optical sensing date, or None.

    Sentinel-1 and Sentinel-2 do not share an orbit, so their P30D aggregation windows close on
    different days. Pairing by nearest date within a tolerance is the join; requiring an exact match
    would discard almost everything.

    `max_gap_days` is deliberately under the 30-day aggregation interval: a radar window centred
    more than 20 days from the optical one describes a different point in the crop cycle.
    """
    if not sar:
        return None
    t = date.fromisoformat(target)
    best, best_gap = None, max_gap_days + 1
    for obs in sar:
        try:
            gap = abs((date.fromisoformat(obs["sensing_date"]) - t).days)
        except ValueError:
            continue
        if gap < best_gap:
            best, best_gap = obs, gap
    return best


def build(daily: dict, sat: dict, site: dict, crop: str = agronomy.DEFAULT_CROP) -> dict:
    """One feature row.

    `daily`   — output of `data.meteo.daily_frame`, ending on the prediction date.
    `sat`     — {ndvi, ndmi, evi, vci, ndvi_z_peer} for the prediction date, optionally merged with
                a `radar_block` ({rvi, vh_vv_ratio, rvi_z_peer}). The radar keys default to NaN when
                absent, so a caller with no Sentinel-1 coverage passes the optical block unchanged
                and still gets a complete, correctly-shaped row.
    `site`    — {latitude, longitude, elevation, clim_rain_30}, the last being the site's own
                climatological 30-day rainfall for this time of year, used for a *real* anomaly.
    """
    nan = float("nan")
    w30, w90 = _slice(daily, 30), _slice(daily, WINDOW_DAYS)

    tmax90, tmin90 = w90["temperature_2m_max"], w90["temperature_2m_min"]
    rain90, et090 = w90["precipitation_sum"], w90["et0_fao_evapotranspiration"]
    tmax30, tmin30 = w30["temperature_2m_max"], w30["temperature_2m_min"]
    rain30, et030 = w30["precipitation_sum"], w30["et0_fao_evapotranspiration"]

    gdd_daily = [agronomy.growing_degree_days(lo, hi) for lo, hi in zip(tmin90, tmax90, strict=True)]

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
        "rvi": _round_or_nan(sat.get("rvi", nan), 4),
        "vh_vv_ratio": _round_or_nan(sat.get("vh_vv_ratio", nan), 4),
        "rvi_z_peer": _round_or_nan(sat.get("rvi_z_peer", nan), 3),

        "latitude": site["latitude"],
        "longitude": site["longitude"],
        "elevation": site.get("elevation", 0.0),
    }


def peer_z(value: float | None, peer_mu_sigma: list[float] | None) -> float:
    """Standardise against the shared reference, or NaN when the bucket was never measured.

    `sigma > 1e-9` mirrors `cluster_relative_row`'s constant-column rule, so the two mechanisms
    cannot disagree about a degenerate reference.
    """
    if peer_mu_sigma is None or value is None or not math.isfinite(value):
        return float("nan")
    mu, sigma = peer_mu_sigma
    if not math.isfinite(sigma) or sigma <= 1e-9:
        return 0.0
    return (value - mu) / sigma


def satellite_block(current: dict, past_ndvi: list[float],
                    peer_mu_sigma: list[float] | None) -> dict:
    """Canopy-state features.

    `past_ndvi` is this site's observations strictly *before* now, so VCI carries no future
    information. `peer_mu_sigma` is the `[mu, sigma]` of the (cluster, calendar-month) reference
    bucket — looked up by the caller from `peer_stats`, never computed here.

    That argument used to be a peer *list*, and the two callers passed different lists: training the
    concurrent cross-site cohort, serving the site's own 12-month history. One column name, two
    meanings, correlated 0.626 with 25.5% sign flips. Taking the reference instead of the sample is
    what makes the two callers incapable of disagreeing.
    """
    ndvi = current["ndvi"]
    history = past_ndvi or [ndvi]
    return {
        "ndvi": ndvi,
        "ndmi": current["ndwi"],   # the repo's "ndwi" is the B08/B11 formula, i.e. NDMI
        "evi": current["evi"],
        "vci": indices.vci(ndvi, min(history), max(history)),
        "ndvi_z_peer": peer_z(ndvi, peer_mu_sigma),
    }


def radar_block(current: dict | None, peer_mu_sigma: list[float] | None) -> dict:
    """Canopy-structure features from Sentinel-1 backscatter.

        RVI      = 4 * VH / (VV + VH)      computed upstream in the evalscript
        VH/VV    = VH / VV                 co- to cross-polarised ratio, rises with canopy structure

    `current` is None when the site has no radar observation within tolerance of the optical date —
    a real and common case, so every field is None-safe and yields NaN rather than a substituted
    value. An invented backscatter is worse than an absent one for the same reason an invented NDVI
    is: `HistGradientBoostingClassifier` routes NaN down its own branch, and a fabricated number
    silently claims the field was observed.

    RVI complements NDVI rather than duplicating it: NDVI saturates in dense canopy and reads
    greenness, while volume scattering tracks structure and biomass through cloud.
    """
    if not current:
        return {"rvi": float("nan"), "vh_vv_ratio": float("nan"), "rvi_z_peer": float("nan")}

    vv, vh, rvi = current["vv"], current["vh"], current["rvi"]
    return {
        "rvi": rvi,
        "vh_vv_ratio": (vh / vv) if vv else float("nan"),
        "rvi_z_peer": peer_z(rvi, peer_mu_sigma),
    }
