"""Open-Meteo access: ERA5 archive for training, forecast for serving.

Both endpoints are queried at *daily* resolution with the same variable list, so a feature computed
here for training is computed from the identical quantity at serving time. That symmetry is the
point — the previous pipeline built `rainfall_anomaly` from a six-month archive total at training
and an eight-day forecast total at serving, under one column name.

Archive responses are cached on disk: a training run touches a few hundred coordinates and reruns
should not re-hit a free public API.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import requests

# Daily ERA5 variables. Kept in one place because the archive and forecast endpoints must agree.
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "relative_humidity_2m_mean",
    "shortwave_radiation_sum",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CACHE_DIR = Path(".cache/meteo")


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.json"


def fetch_archive(lat: float, lon: float, start: str, end: str, timeout: int = 60) -> dict | None:
    """Daily ERA5 reanalysis for a coordinate over [start, end] (ISO dates). Cached on disk."""
    key = f"{lat:.4f},{lon:.4f},{start},{end}"
    path = _cache_path(key)
    if path.exists():
        return json.loads(path.read_text())

    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_VARS), "timezone": "UTC",
    }
    # The archive endpoint is the expensive one and rate-limits a parallel backfill hard. Backing
    # off is cheaper than shrinking the dataset to whatever survived the first attempt.
    for attempt in range(5):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(2 ** attempt + random.random())
                continue
            resp.raise_for_status()
            payload = resp.json()
            if "daily" not in payload:
                return None
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
            return payload
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                print(f"[meteo] archive {lat:.3f},{lon:.3f} failed: {e}")
                return None
            time.sleep(2 ** attempt + random.random())
    return None


def fetch_recent(lat: float, lon: float, past_days: int = 92, timeout: int = 8) -> dict | None:
    """Daily observations for the last `past_days`, from the forecast endpoint.

    This is the serving-side counterpart of `fetch_archive`: same variables, same daily resolution,
    so the 90-day feature window is built from the same quantities in both paths. 92 days is the
    endpoint's maximum look-back.
    """
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "past_days": min(past_days, 92), "forecast_days": 1, "timezone": "UTC",
    }
    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        return payload if "daily" in payload else None
    except Exception as e:  # noqa: BLE001
        print(f"[meteo] forecast {lat:.3f},{lon:.3f} failed: {e}")
        return None


def climatological_rain_30(lat: float, lon: float, target_mmdd: str, years: int = 4) -> float | None:
    """Mean 30-day rainfall ending on `target_mmdd` (MM-DD) across the last `years` at this point.

    Serving needs the same reference the training set used, or `rain_anomaly_30` means two different
    things again. One archive call per coordinate, disk-cached, so only the first prediction for a
    site pays for it.
    """
    from datetime import date, timedelta

    end = date.today() - timedelta(days=6)          # ERA5 lags ~5 days
    payload = fetch_archive(lat, lon, (end - timedelta(days=years * 365)).isoformat(), end.isoformat())
    if not payload:
        return None
    daily = daily_frame(payload)
    totals = [sum(daily["precipitation_sum"][i - 30:i])
              for i, ts in enumerate(daily["time"]) if i >= 30 and ts[5:] == target_mmdd]
    return sum(totals) / len(totals) if totals else None


def daily_frame(payload: dict) -> dict[str, list]:
    """Pull the daily block out of either endpoint's response, with None gaps filled forward.

    ERA5 has occasional single-day gaps; a None in the middle of a rainfall series would otherwise
    propagate a TypeError into every downstream sum.
    """
    daily = payload["daily"]
    out = {"time": daily["time"]}
    for var in DAILY_VARS:
        series = daily.get(var) or [None] * len(daily["time"])
        filled, last = [], 0.0
        for v in series:
            last = float(v) if v is not None else last
            filled.append(last)
        out[var] = filled
    return out
