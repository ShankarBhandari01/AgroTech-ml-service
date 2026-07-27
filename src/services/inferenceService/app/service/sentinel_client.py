"""
Real Sentinel-2 spectral indices for coldstart inference, via the Copernicus Data Space Ecosystem
(CDSE) Sentinel Hub **Statistical API**. The Statistical API returns aggregated band statistics
(mean NDVI/NDWI/EVI over a small AOI + time window) as JSON — no raster/GeoTIFF decoding needed,
which is why it's the right tool for *feature* extraction (the map-tile visualisation path lives in
the Kotlin backend instead).

Disabled (returns None) when no credentials are configured, so PredictionsService transparently
falls back to its synthetic index model. Blocking `requests` calls are used deliberately — the
caller wraps this in `run_in_threadpool` so the event loop isn't blocked.
"""
# Defers annotation evaluation so PEP 604 unions (`dict | None`) work on this service's Python 3.9.
from __future__ import annotations

import math
import threading
import time
from datetime import date, timedelta

import requests

from src.services.inferenceService.app.core.config import settings

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
        if not self.enabled or lat is None or lon is None:
            return None
        try:
            token = self._access_token()
            d = 0.005  # ~500 m AOI half-width
            end = date.today()
            start = end - timedelta(days=30)
            body = {
                "input": {
                    "bounds": {
                        "bbox": [lon - d, lat - d, lon + d, lat + d],
                        "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                    },
                    "data": [{"type": "sentinel-2-l2a", "dataFilter": {"maxCloudCoverage": 40}}],
                },
                "aggregation": {
                    "timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                    "aggregationInterval": {"of": "P30D"},
                    "evalscript": _EVALSCRIPT,
                    "resx": 10,
                    "resy": 10,
                },
            }
            resp = requests.post(
                settings.SENTINEL_STATS_URL,
                json=body,
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
            )
            resp.raise_for_status()
            intervals = resp.json().get("data", [])
            # Most recent interval that actually has valid stats (skip fully cloud-masked windows).
            for interval in reversed(intervals):
                outputs = interval.get("outputs", {})
                ndvi = self._mean(outputs, "ndvi")
                ndwi = self._mean(outputs, "ndwi")
                evi = self._mean(outputs, "evi")
                if ndvi is not None and ndwi is not None and evi is not None:
                    return {
                        "ndvi": ndvi,
                        "ndwi": ndwi,
                        "evi": evi,
                        "sensing_date": str(interval.get("interval", {}).get("to", ""))[:10],
                    }
            return None
        except Exception as e:  # noqa: BLE001 — any failure degrades to synthetic, never breaks inference
            print(f"[SentinelClient] fetch_indices failed: {e}")
            return None

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
