"""
Real Sentinel-2 spectral indices for coldstart inference, via the Copernicus Data Space Ecosystem
(CDSE) Sentinel Hub **Statistical API**. The Statistical API returns aggregated band statistics
(mean NDVI/NDWI/EVI over a small AOI + time window) as JSON — no raster/GeoTIFF decoding needed,
which is why it's the right tool for *feature* extraction (the map-tile visualisation path lives in
the Kotlin backend instead).

Returns nothing when no credentials are configured or no cloud-free scene exists. Callers mark the
canopy block unavailable rather than substituting modelled indices: an invented NDVI is worse than
an absent one, because the physics-derived hazards carry the assessment perfectly well without it.

Blocking `requests` calls are used deliberately — callers wrap this in `run_in_threadpool` so the
event loop isn't blocked.
"""
from __future__ import annotations

import math
import random
import threading
import time
from datetime import date, timedelta

import requests

from argotech.config import settings

# Aggregated NDVI / NDWI / EVI over the AOI. dataMask output lets the Statistical API exclude
# no-data/cloud-masked pixels from the mean. EVI uses the blue band (B02) per the standard formula.
_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B04", "B08", "B11", "dataMask"] }],
    output: [
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "ndwi", bands: 1, sampleType: "FLOAT32" },
      { id: "evi",  bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(s) {
  let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04);
  let ndwi = (s.B08 - s.B11) / (s.B08 + s.B11);
  let evi  = 2.5 * (s.B08 - s.B04) / (s.B08 + 6.0 * s.B04 - 7.5 * s.B02 + 1.0);
  return { ndvi: [ndvi], ndwi: [ndwi], evi: [evi], dataMask: [s.dataMask] };
}
"""

# Raw Sentinel-2 L2A band means, in the exact set Presto was pre-trained on (B1/B9/B10 are dropped
# upstream by Presto itself, so they are not requested here). This is deliberately *separate* from
# `_EVALSCRIPT`: the index evalscript is what the serving path and the tabular features use, and
# reusing one script for both would couple a modelling experiment to the production feature cache.
#
# Values come back as L2A surface reflectance in [0, 1]. Presto normalises Earth Engine's 0-10000
# integers by dividing by 1e4, so CDSE's float reflectance is already on Presto's scale and needs no
# further scaling — see `embeddings.build_input`.
_S2_BANDS_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","dataMask"] }],
    output: [
      { id: "b02", bands: 1, sampleType: "FLOAT32" },
      { id: "b03", bands: 1, sampleType: "FLOAT32" },
      { id: "b04", bands: 1, sampleType: "FLOAT32" },
      { id: "b05", bands: 1, sampleType: "FLOAT32" },
      { id: "b06", bands: 1, sampleType: "FLOAT32" },
      { id: "b07", bands: 1, sampleType: "FLOAT32" },
      { id: "b08", bands: 1, sampleType: "FLOAT32" },
      { id: "b8a", bands: 1, sampleType: "FLOAT32" },
      { id: "b11", bands: 1, sampleType: "FLOAT32" },
      { id: "b12", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(s) {
  return {
    b02: [s.B02], b03: [s.B03], b04: [s.B04], b05: [s.B05], b06: [s.B06],
    b07: [s.B07], b08: [s.B08], b8a: [s.B8A], b11: [s.B11], b12: [s.B12],
    dataMask: [s.dataMask]
  };
}
"""

S2_BAND_KEYS = ("b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b11", "b12")

# Sentinel-1 C-band backscatter. Radar sees through cloud, which is the entire reason this exists:
# optical gaps cluster in exactly the rainy months when a drought or disease signal matters most.
#
#   RVI  = 4 * VH / (VV + VH)        Radar Vegetation Index, ~0 for bare soil, ~1 for dense canopy.
#                                    Sensitive to canopy volume scattering, so it tracks biomass
#                                    where NDVI tracks greenness — related but not redundant.
#   VH/VV                            Cross- to co-polarised ratio; rises with vegetation structure
#                                    and is less sensitive to incidence-angle geometry than VH alone.
#
# Linear power units (not dB) so the means are physically meaningful before averaging: averaging
# decibels averages logarithms, which is not the mean backscatter.
_S1_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VV", "VH", "dataMask"] }],
    output: [
      { id: "vv",  bands: 1, sampleType: "FLOAT32" },
      { id: "vh",  bands: 1, sampleType: "FLOAT32" },
      { id: "rvi", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(s) {
  let denom = s.VV + s.VH;
  let rvi = denom > 0 ? (4.0 * s.VH) / denom : 0;
  return { vv: [s.VV], vh: [s.VH], rvi: [rvi], dataMask: [s.dataMask] };
}
"""


# Normalised-difference indices are mathematically bounded to [-1, 1]. EVI is not a normalised
# difference: its denominator `NIR + 6*Red - 7.5*Blue + 1` goes NEGATIVE when blue reflectance is
# high (haze, thin cloud, failed atmospheric correction), and the evalscript ratio then blows up.
# Measured on this repository's band cache (6,925 observations, 165 sites): 3.08% of denominators
# are negative and 4.03% of EVI values fall outside [-1, 1], reaching 82.4. `data/training_set.parquet`
# carries the same contamination -- 2.63% of rows, range -32.4 .. 82.4 -- so every model trained on
# `evi` has been fed values two orders of magnitude out of range.
#
# The out-of-range value is set to NaN, not clipped: with a negative denominator over a green canopy
# the ratio is negative, so clipping would report a cloudy pixel over a HEALTHY field as -1, "worst
# possible vegetation". That is the same failure `data/meteo.py` records for a missing ET0 read as
# zero. Only the offending index is NaN'd -- the interval is kept, because dropping it would discard
# a perfectly good NDVI along with the bad EVI.
_INDEX_RANGE = {"ndvi": (-1.0, 1.0), "ndwi": (-1.0, 1.0), "ndmi": (-1.0, 1.0), "evi": (-1.0, 1.0)}


def _validate_index(key: str, value):
    lo_hi = _INDEX_RANGE.get(key)
    if lo_hi is None or value is None:
        return value
    lo, hi = lo_hi
    return value if lo <= value <= hi else float("nan")


class SentinelClient:
    def __init__(self) -> None:
        self._token: str | None = None
        self._token_exp: float = 0.0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(settings.SENTINEL_CLIENT_ID and settings.SENTINEL_CLIENT_SECRET)

    def _access_token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._token_exp:
                return self._token
            resp = requests.post(
                settings.SENTINEL_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": settings.SENTINEL_CLIENT_ID,
                    "client_secret": settings.SENTINEL_CLIENT_SECRET,
                },
                timeout=10,
            )
            resp.raise_for_status()
            js = resp.json()
            self._token = js["access_token"]
            # Refresh a minute early; the CDSE token endpoint is rate-limited.
            self._token_exp = time.time() + max(30, int(js.get("expires_in", 3600)) - 60)
            return self._token

    def fetch_indices(self, lat, lon) -> dict | None:
        """Mean NDVI/NDWI/EVI at (lat, lon) over the last ~30 days, or None if unavailable."""
        for obs in reversed(self._stats(lat, lon, days=30, interval="P30D")):
            return obs  # already filtered to intervals with valid stats
        return None

    def fetch_history(self, lat, lon, days: int = 365, res_m: int = 10) -> list[dict]:
        """Monthly index observations for the last `days`, oldest first.

        This is what turns a raw NDVI into a meaningful one: VCI and the peer anomaly in
        `domain.indices` both need a reference distribution, and one extra Statistical API call
        buys the field's own 12-month history instead of a hard-coded regional prior.
        """
        return self._stats(lat, lon, days=days, interval="P30D", res_m=res_m)

    def fetch_sar_history(self, lat, lon, days: int = 365, res_m: int = 20) -> list[dict]:
        """Monthly Sentinel-1 backscatter observations for the last `days`, oldest first.

        The cloud-gap fix. Optical coverage fails in the rainy season — precisely when a water-stress
        or disease signal is most actionable — and radar does not care about cloud, so this returns a
        canopy-structure observation for intervals where `fetch_history` returns nothing.

        20 m default resolution rather than 10: S1 GRD is natively ~10x10 m ground range but speckle
        makes single-pixel values noisy, and averaging over a coarser grid is the standard mitigation.
        """
        return self._stats(lat, lon, days=days, interval="P30D", res_m=res_m,
                           collection="sentinel-1-grd", evalscript=_S1_EVALSCRIPT,
                           keys=("vv", "vh", "rvi"), data_filter={})

    def fetch_bands_history(self, lat, lon, days: int = 365, res_m: int = 60) -> list[dict]:
        """Monthly raw Sentinel-2 band means, for the Presto embedding path only.

        Not used by the serving features or the tabular training set — those consume the derived
        indices from `fetch_history`. This exists because Presto's pre-training expects reflectance
        in ten named bands, and an index cannot be inverted back into them.
        """
        return self._stats(lat, lon, days=days, interval="P30D", res_m=res_m,
                           evalscript=_S2_BANDS_EVALSCRIPT, keys=S2_BAND_KEYS)

    def _stats(self, lat, lon, days: int, interval: str, res_m: int = 10,
               collection: str = "sentinel-2-l2a", evalscript: str = _EVALSCRIPT,
               keys: tuple[str, ...] = ("ndvi", "ndwi", "evi"),
               data_filter: dict | None = None) -> list[dict]:
        """Aggregated statistics per interval, oldest first. Empty list on any failure.

        Parameterised over collection/evalscript/outputs so Sentinel-1 and Sentinel-2 share one
        request path, one token, one error contract and one retry story. The alternative — a second
        near-identical method — is where the two silently drift apart.
        """
        if not self.enabled or lat is None or lon is None:
            return []
        try:
            token = self._access_token()
            d = 0.005  # ~500 m AOI half-width
            end = date.today()
            start = end - timedelta(days=days)
            if data_filter is None:
                data_filter = {"maxCloudCoverage": 40}
            body = {
                "input": {
                    "bounds": {
                        "bbox": [lon - d, lat - d, lon + d, lat + d],
                        "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                    },
                    "data": [{"type": collection, "dataFilter": data_filter}],
                },
                "aggregation": {
                    "timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                    "aggregationInterval": {"of": interval},
                    "evalscript": evalscript,
                    "resx": res_m,
                    "resy": res_m,
                },
            }
            # Retry on 429. A bulk backfill (165 sites x 4 years x 10 bands) hits the CDSE rate
            # limit reliably, and without this the request fails, returns [], and the caller
            # caches the empty list as though the site genuinely had no imagery. That poisoned
            # 42% of a band fetch — concentrated in whole clusters, which is exactly the pattern
            # that corrupts leave-one-cluster-out. Serving is unaffected: it makes one request at
            # a time and never retries more than a request's own latency budget.
            for attempt in range(5):
                resp = requests.post(
                    settings.SENTINEL_STATS_URL,
                    json=body,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=15,
                )
                if resp.status_code != 429:
                    break
                time.sleep(2 ** attempt + random.random())
            resp.raise_for_status()
            out = []
            # Keep only intervals that actually have valid stats (skip fully cloud-masked windows).
            for iv in resp.json().get("data", []):
                outputs = iv.get("outputs", {})
                values = {k: self._mean(outputs, k) for k in keys}
                if all(v is not None for v in values.values()):
                    values = {k: _validate_index(k, v) for k, v in values.items()}
                    values["sensing_date"] = str(iv.get("interval", {}).get("to", ""))[:10]
                    out.append(values)
            return out
        except Exception as e:  # noqa: BLE001 — any failure degrades to physics-only, never breaks inference
            print(f"[SentinelClient] statistics query failed: {e}")
            return []

    @staticmethod
    def _mean(outputs: dict, key: str):
        try:
            stats = outputs[key]["bands"]["B0"]["stats"]
            if stats.get("sampleCount", 0) == 0:
                return None
            mean = stats.get("mean")
            if mean is None or (isinstance(mean, float) and math.isnan(mean)):
                return None
            return float(mean)
        except (KeyError, TypeError):
            return None


# Module-level singleton so the OAuth token is cached across requests.
sentinel_client = SentinelClient()
