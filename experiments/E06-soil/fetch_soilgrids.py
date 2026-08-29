"""Fetch SoilGrids v2.0 properties for the 122 sites with an estimable field effect.

Cache: `.cache/soilgrids/{site_id}-{property}.json`, no date in the key (soil is static) — same
convention `lab.panel.panel._cached_fetch` uses for its own `{site_id}-{days}.json` keys. One file
per (site, property): a crash mid-run keeps everything already fetched.

Requests are grouped (2 per site: a fast 3-property group and a slower 4-property group, both
depths at once) rather than one call per property — probing before this run showed the API is slow
and unreliable per request regardless of size (single-property both-depths calls measured 2-30s,
one outright timeout), so cutting the round-trip count matters more than shrinking each payload.
Each grouped response is split back into one cache file per property, so the fine-grained cache
above still holds.

Refuses to cache an empty/failed response, exactly as `_cached_fetch` does and for the same reason:
memoizing a transient failure turns it into a permanent hole (see panel.py's own comment on the 69
band-less sites this caused for Sentinel).

Run: venv/bin/python3 experiments/E06-soil/fetch_soilgrids.py
"""
from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO / ".cache" / "soilgrids"
SITES_CSV = Path("/tmp/e06_sites.csv")

URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
# Split into a fast and a slow group (measured while probing: group A ~10s, group B ~20s per site
# for both depths) rather than one 7-property call, which timed out outright at 90s.
GROUPS = [["clay", "sand", "silt"], ["soc", "phh2o", "cec", "bdod"]]
DEPTHS = ["0-5cm", "5-15cm"]
WORKERS = 6
TIMEOUT = 60
RETRIES = 3


def _cache_path(site_id: str, prop: str) -> Path:
    return CACHE_DIR / f"{site_id}-{prop}.json"


def fetch_group(site_id: str, lat: float, lon: float, group: list[str]) -> tuple[str, int]:
    """Fetch one property group for one site, both depths. Writes one cache file per property that
    came back complete. Returns (site_id, n_properties_cached)."""
    missing = [p for p in group if not _cache_path(site_id, p).exists()]
    if not missing:
        return site_id, 0

    params = [("lon", lon), ("lat", lat)] + [("property", p) for p in missing] \
        + [("depth", d) for d in DEPTHS] + [("value", "mean")]
    for attempt in range(RETRIES):
        try:
            resp = requests.get(URL, params=params, timeout=TIMEOUT)
            if resp.status_code != 200:
                time.sleep(2 ** attempt + random.random())
                continue
            data = resp.json()
            layers = data.get("properties", {}).get("layers", [])
            written = 0
            for prop in missing:
                layer = next((lyr for lyr in layers if lyr["name"] == prop), None)
                # Refuse to memoise a missing/empty layer or one short a depth — an upstream miss,
                # not a soil reading of zero.
                if not layer or len(layer.get("depths", [])) < len(DEPTHS):
                    continue
                path = _cache_path(site_id, prop)
                path.parent.mkdir(parents=True, exist_ok=True)
                # temp+rename: this cache dir is shared with lab/covariates/site_covariates.py,
                # whose reader would otherwise observe a half-written file. build_and_regress.py:51
                # reads it with a bare json.loads and has no such guard.
                tmp = path.with_suffix(f".tmp{os.getpid()}")
                tmp.write_text(json.dumps(layer))
                tmp.rename(path)
                written += 1
            if written == len(missing):
                return site_id, written
            time.sleep(2 ** attempt + random.random())
        except (requests.RequestException, ValueError):
            time.sleep(2 ** attempt + random.random())
    # Return however many properties did get cached even on eventual failure/timeout — a partial
    # group result is still real cached data, not memoised failure.
    return site_id, sum(1 for p in group if _cache_path(site_id, p).exists())


def main() -> None:
    sites = pd.read_csv(SITES_CSV)
    jobs = [(row.site_id, row.latitude, row.longitude, group)
            for row in sites.itertuples() for group in GROUPS]
    total_props = len(sites) * sum(len(g) for g in GROUPS)
    done_jobs = cached_props = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch_group, *j) for j in jobs]
        for fut in as_completed(futures):
            fut.result()
            done_jobs += 1
            cached_props = sum(1 for _ in CACHE_DIR.glob("*.json"))
            if done_jobs % 10 == 0 or done_jobs == len(jobs):
                elapsed = time.time() - t0
                print(f"[{done_jobs}/{len(jobs)} requests] cached={cached_props}/{total_props} "
                      f"properties elapsed={elapsed:.0f}s", flush=True)

    print(f"DONE {cached_props}/{total_props} properties cached, {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
