"""Build `data/site_covariates.parquet` — static, site-keyed soil and terrain covariates.

Why this is a separate file, not columns bolted onto `data/training_set.parquet`
----------------------------------------------------------------------------
Appending columns there would change its `content_hash`, and 38 committed experiment result files
pin that hash in their provenance blocks. Changing it would make every one of them name a dataset
that no longer exists — precisely the failure the provenance system exists to prevent. So this is a
separate, site-keyed file that experiments opt into by joining on `site_id`.

Sources
-------
* **Soil** — SoilGrids v2.0 (ISRIC), 250 m, `clay`/`sand`/`silt`/`soc`/`phh2o`/`cec`/`bdod` at
  0-5cm and 5-15cm, `value=mean`. The API returns scaled integers (e.g. `clay=159` for a real Kaduna
  site means 15.9%); converted with the response's own `unit_measure.d_factor`, never a hardcoded
  divisor, because the scale factor is not the same for every property in principle.
* **Slope** — not upstream anywhere: `lab/presto/embeddings.py:170` sets every site's slope to a
  fabricated `0.0` ("no slope upstream; 0 == flat after /50"). Computed here from Open-Meteo's
  elevation field (the same source already read at `elevation = payload["elevation"]` in
  `lab/panel/panel.py:230`), sampled at four points ±0.005° around the site — the same half-width as
  the Sentinel bounding box (`data/sentinel.py:204`) — via the dedicated batched elevation endpoint,
  and reported as a percent-grade gradient magnitude (never wired into `embeddings.py` — that is a
  separate change with its own consequences).

Sites come from the panel (`data/training_set.parquet`'s `site_id`/`latitude`/`longitude`, 122 of
them), not re-derived from `sample_sites`.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests

TRAINING_SET = Path("data/training_set.parquet")
OUT_PATH = Path("data/site_covariates.parquet")

# Shared with `lab/panel/panel.py`'s Sentinel/SAR caches and E06's SoilGrids fetch: this directory
# is read by more than one concurrent process, so entries are never rewritten or deleted here, and
# writes go through a temp-file-then-rename so a half-written file is never observed by a reader.
CACHE_DIR = Path(".cache/soilgrids")

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOILGRIDS_PROPERTIES = ["clay", "sand", "silt", "soc", "phh2o", "cec", "bdod"]
SOILGRIDS_DEPTHS = ["0-5cm", "5-15cm"]

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
# Same half-width as the Sentinel-2 AOI bbox (`src/argotech/data/sentinel.py:204`), so slope is
# measured over the same footprint the canopy indices are aggregated over.
SLOPE_HALF_WIDTH_DEG = 0.005
METERS_PER_DEG_LAT = 111_320.0


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write via temp file + rename so a concurrent reader never observes a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    tmp.rename(path)


def _cached_fetch(path: Path, fetch) -> dict | None:
    """Read-through cache that refuses to memoise a failure.

    Same rule as `lab/panel/panel.py:_cached_fetch`: an indistinguishable empty/failed response,
    once cached, silently and permanently blinds every site that hit it. Not writing costs a retry
    next run; writing costs a silently corrupted dataset. Never deletes or overwrites an existing
    entry — other agents share this cache directory.
    """
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            # Another writer's temp file was observed mid-rename (should not happen with the
            # temp+rename scheme above, but a reader must not crash on it) — treat as a miss.
            return None
    obs = fetch()
    if not obs:
        return obs
    _atomic_write_json(path, obs)
    return obs


def fetch_soilgrids_property(lat: float, lon: float, prop: str, timeout: int = 90) -> dict | None:
    """One SoilGrids v2.0 properties/query call, scoped to a single property (both depths).

    Cache keys are per-property (`{site_id}-{property}.json`) rather than one combined file per
    site: that is the shape the concurrent SoilGrids fetch (E06's experiment) already writes into
    this same directory, so keying identically means a property either of us has already fetched
    is read, not re-requested — the point of "share that cache". Returns the single `layers[0]`
    dict, never an empty one.
    """
    params = {"lon": lon, "lat": lat, "property": prop, "depth": SOILGRIDS_DEPTHS, "value": "mean"}
    for attempt in range(4):
        try:
            resp = requests.get(SOILGRIDS_URL, params=params, timeout=timeout)
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt + random.random())
                continue
            resp.raise_for_status()
            layers = resp.json().get("properties", {}).get("layers")
            if not layers:
                return None
            return layers[0]
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"[soilgrids] {prop} {lat:.4f},{lon:.4f} failed: {e}")
                return None
            time.sleep(2 ** attempt + random.random())
    return None


def layer_values(layer: dict) -> dict[str, float]:
    """Flatten one SoilGrids property layer into `{property}_{depth}` -> converted value.

    Conversion uses the layer's own `unit_measure.d_factor` (the API's scale factor), never a
    hardcoded divisor: `value_mean / d_factor` turns the returned scaled integer (e.g. `clay=159`)
    into the property's natural unit (15.9 %).
    """
    name = layer["name"]
    d_factor = layer["unit_measure"]["d_factor"]
    out: dict[str, float] = {}
    for depth in layer["depths"]:
        mean = depth["values"].get("mean")
        if mean is None:
            continue
        out[f"{name}_{depth['label']}"] = mean / d_factor
    return out


def soil_row(site_id: str, lat: float, lon: float) -> dict[str, float] | None:
    """All soil properties for one site, each read through the shared per-property cache.

    `None` only when every property failed; a partial response contributes whatever properties did
    come back (never a fabricated value for the ones that didn't) — the per-site refusal-to-fabricate
    rule applied at property grain, since one flaky property should not blank out six good ones.
    """
    out: dict[str, float] = {}
    for prop in SOILGRIDS_PROPERTIES:
        layer = _cached_fetch(
            CACHE_DIR / f"{site_id}-{prop}.json",
            lambda lat=lat, lon=lon, prop=prop: fetch_soilgrids_property(lat, lon, prop),
        )
        if layer:
            out.update(layer_values(layer))
    return out or None


def fetch_elevation_grid(lat: float, lon: float, timeout: int = 30) -> list[float] | None:
    """Elevation at [N, S, E, W] points ±`SLOPE_HALF_WIDTH_DEG` around (lat, lon), one HTTP call.

    Same Open-Meteo host already used for weather (`data/meteo.py`); its forecast/archive responses
    carry an `elevation` field for the query point (`lab/panel/panel.py:230`) — this is the same
    provider's dedicated batched elevation endpoint, not a new dependency or a new provider.
    """
    d = SLOPE_HALF_WIDTH_DEG
    lats = [lat + d, lat - d, lat, lat]
    lons = [lon, lon, lon + d, lon - d]
    params = {"latitude": ",".join(f"{v:.6f}" for v in lats),
              "longitude": ",".join(f"{v:.6f}" for v in lons)}
    for attempt in range(4):
        try:
            resp = requests.get(ELEVATION_URL, params=params, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(2 ** attempt + random.random())
                continue
            resp.raise_for_status()
            elevs = resp.json().get("elevation")
            if not elevs or len(elevs) != 4 or any(e is None for e in elevs):
                return None
            return [float(e) for e in elevs]
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"[elevation] {lat:.4f},{lon:.4f} failed: {e}")
                return None
            time.sleep(2 ** attempt + random.random())
    return None


def slope_percent(lat: float, elevs: list[float]) -> float:
    """Gradient magnitude, as **percent grade** (rise/run x 100), from a [N, S, E, W] elevation grid.

    Central difference over the 2 x `SLOPE_HALF_WIDTH_DEG` span, converted from degrees to meters:
    1 deg latitude = 111,320 m everywhere; 1 deg longitude = 111,320 m x cos(latitude).
    """
    n, s, e, w = elevs
    span_lat_m = 2 * SLOPE_HALF_WIDTH_DEG * METERS_PER_DEG_LAT
    span_lon_m = 2 * SLOPE_HALF_WIDTH_DEG * METERS_PER_DEG_LAT * math.cos(math.radians(lat))
    dz_dlat = (n - s) / span_lat_m
    dz_dlon = (e - w) / span_lon_m if span_lon_m else 0.0
    return math.hypot(dz_dlat, dz_dlon) * 100.0


def load_sites() -> pd.DataFrame:
    """The 122 sites from the panel — never re-derived from `sample_sites`."""
    df = pd.read_parquet(TRAINING_SET, columns=["site_id", "latitude", "longitude", "cluster"])
    return df.drop_duplicates("site_id").reset_index(drop=True)


FULL_SOIL_COLUMNS = len(SOILGRIDS_PROPERTIES) * len(SOILGRIDS_DEPTHS)  # 14: 7 properties x 2 depths


def build(sites: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per site. A site with no response contributes no row for that covariate group —
    a hole, never a fabricated zero. Returns (frame, coverage_report)."""
    rows = []
    soil_hits, soil_full_hits, slope_hits = 0, 0, 0
    soil_miss_clusters: dict[str, int] = {}
    slope_miss_clusters: dict[str, int] = {}

    for _, site in sites.iterrows():
        site_id, lat, lon, cluster = site["site_id"], site["latitude"], site["longitude"], site["cluster"]
        row: dict = {"site_id": site_id, "cluster": cluster}

        soil = soil_row(site_id, lat, lon)
        if soil:
            row.update(soil)
            soil_hits += 1
            if len(soil) == FULL_SOIL_COLUMNS:
                soil_full_hits += 1
        else:
            soil_miss_clusters[cluster] = soil_miss_clusters.get(cluster, 0) + 1

        elev_payload = _cached_fetch(
            CACHE_DIR / f"{site_id}-slope.json",
            lambda lat=lat, lon=lon: (
                {"elevation": e} if (e := fetch_elevation_grid(lat, lon)) else None
            ),
        )
        if elev_payload:
            row["slope_percent"] = slope_percent(lat, elev_payload["elevation"])
            slope_hits += 1
        else:
            slope_miss_clusters[cluster] = slope_miss_clusters.get(cluster, 0) + 1

        rows.append(row)

    frame = pd.DataFrame(rows)
    n = len(sites)
    cluster_sizes = sites.groupby("cluster")["site_id"].count().to_dict()
    report = {
        "n_sites": n,
        "soil_hits": soil_hits,              # sites with >=1 of 7 properties
        "soil_full_hits": soil_full_hits,    # sites with all 7 properties x 2 depths
        "slope_hits": slope_hits,
        "soil_miss_clusters": soil_miss_clusters,
        "slope_miss_clusters": slope_miss_clusters,
        "cluster_sizes": cluster_sizes,
    }
    return frame, report


def _cluster_structured(miss_by_cluster: dict[str, int], cluster_sizes: dict[str, int]) -> bool:
    """True when misses land entirely inside a strict subset of clusters — the one shape that
    would invalidate leave-one-cluster-out, since it makes 'held-out cluster' and 'no covariate'
    the same event."""
    if not miss_by_cluster:
        return False
    full_clusters_missing = [c for c, n in miss_by_cluster.items() if n == cluster_sizes.get(c, -1)]
    return len(full_clusters_missing) > 0 and len(miss_by_cluster) < len(cluster_sizes)


def print_coverage(report: dict) -> None:
    n = report["n_sites"]
    print(f"[site_covariates] soil: {report['soil_hits']}/{n} sites (>=1 property), "
          f"{report['soil_full_hits']}/{n} with all 7 properties")
    print(f"[site_covariates] slope: {report['slope_hits']}/{n} sites")
    for label, misses in (("soil", report["soil_miss_clusters"]), ("slope", report["slope_miss_clusters"])):
        if not misses:
            continue
        print(f"[site_covariates] {label} gaps by cluster: {misses} (cluster sizes: {report['cluster_sizes']})")
        if _cluster_structured(misses, report["cluster_sizes"]):
            print(f"[site_covariates] WARNING: {label} gaps are cluster-structured — "
                  f"at least one whole cluster has zero coverage. This is the one gap shape that "
                  f"invalidates leave-one-cluster-out for this covariate.")


def main() -> None:
    sites = load_sites()
    frame, report = build(sites)
    print_coverage(report)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    print(f"[site_covariates] wrote {OUT_PATH} ({len(frame)} rows, {len(frame.columns)} columns)")


if __name__ == "__main__":
    main()
