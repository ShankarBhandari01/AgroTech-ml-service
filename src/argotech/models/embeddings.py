"""Frozen Presto embeddings from the upstreams we already cache.

Training-only: this module imports torch at module level, so nothing on the serving path may
import it. See the `train` extra in pyproject.toml.

Presto (arXiv 2304.14065) is a 402K-parameter transformer pre-trained on remote-sensing *pixel
timeseries* — the same shape of data this repo already collects. A frozen encoder plus a default
random forest reached mean F1 0.836 on CropHarvest (Kenya maize, Togo cropland, Brazil coffee)
against a task-specific SOTA of 0.802, which is why it is worth a controlled trial here.

What we can and cannot feed it
------------------------------
Presto's input is 12 monthly timesteps x 17 channels, in 9 maskable band groups. We have four of
the nine outright and must mask the rest:

    group          idx      Presto wants          we have
    S1             0,1      VV, VH                yes (added for the radar work)
    S2_RGB         2,3,4    B2, B3, B4            no  -- cache stores indices, not band means
    S2_Red_Edge    5,6,7    B5, B6, B7            no
    S2_NIR_10m     8        B8                    no
    S2_NIR_20m     9        B8A                   no
    S2_SWIR        10,11    B11, B12              no
    ERA5           12,13    temperature, precip   yes
    SRTM           14,15    elevation, slope      elevation only
    NDVI           16       NDVI                  yes

Masking whole groups is a designed-for case, not a workaround: the pretraining objective masks by
channel group precisely so the model tolerates absent sensors. But this is a *reduced-input* trial,
not a full evaluation of Presto — the reflectance groups carry spectral detail we are withholding.
Feeding them means a new evalscript returning band means and a full re-fetch of every site's optical
history, which is only worth paying for if this cheap version shows signal.

Unit conversions
----------------
Two of these are silent failures — wrong units produce embeddings that look perfectly healthy and
mean nothing, so they are asserted in `tests/test_embeddings.py` rather than trusted:

* **S1** — Presto was trained on Earth Engine's Sentinel-1, which is in **decibels** (normalised
  `(x + 25) / 25`). Our CDSE evalscript deliberately returns **linear power**, so it must be
  converted with `10 * log10(x)` first. Skipping this feeds values near 0.1 where the model expects
  values near -15.
* **Precipitation** — Earth Engine's ERA5 `total_precipitation` is **metres/day** (normalised
  `/ 0.03`). Open-Meteo's `precipitation_sum` is **millimetres/day**. A factor of 1000.

Temperature is Kelvin upstream (`(K - 272.15) / 35`); ours is Celsius, so `(C + 1) / 35` is the same
arithmetic. Elevation divides by 2000, slope by 50 — we have no slope, so it is 0 (flat), which only
enters at t=0 because Presto reduces SRTM to a single token rather than one per timestep.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import torch

from argotech.models._presto_vendored import Presto

# Channel layout, mirrored from the vendored `BANDS_GROUPS_IDX`. Duplicated as explicit names
# because an off-by-one here is invisible: every index is a valid float slot.
NUM_TIMESTEPS = 12
NUM_CHANNELS = 17

IDX_VV, IDX_VH = 0, 1
IDX_ERA5_TEMP, IDX_ERA5_PRECIP = 12, 13
IDX_ELEVATION, IDX_SLOPE = 14, 15
IDX_NDVI = 16

# Every optical reflectance channel, and the CDSE output id that fills it. Order is Presto's, not
# Sentinel's: B8A sits between B8 and B11 because that is where the NIR_20m group lives.
OPTICAL_CHANNELS = list(range(2, 12))
BAND_CHANNELS = {
    2: "b02", 3: "b03", 4: "b04",      # S2_RGB
    5: "b05", 6: "b06", 7: "b07",      # S2_Red_Edge
    8: "b08",                          # S2_NIR_10m
    9: "b8a",                          # S2_NIR_20m
    10: "b11", 11: "b12",              # S2_SWIR
}

# Presto's normalisation constants, from `presto/dataops/pipelines/s1_s2_era5_srtm.py`.
S1_SHIFT, S1_DIV = 25.0, 25.0
ERA5_TEMP_SHIFT, ERA5_TEMP_DIV = -272.15, 35.0
ERA5_PRECIP_DIV = 0.03
ELEVATION_DIV, SLOPE_DIV = 2000.0, 50.0

DYNAMIC_WORLD_MISSING = 9      # the "no class" index the encoder treats as masked
WEIGHTS = Path(".cache/presto/default_model.pt")


def linear_to_db(power: float) -> float:
    """Sentinel-1 backscatter, linear power -> decibels.

    Guards the zero/negative case: CDSE returns 0 for a fully masked AOI, and log10(0) is -inf,
    which would propagate through the encoder as NaN. -50 dB is Earth Engine's documented floor.
    """
    return 10.0 * math.log10(power) if power > 0 else -50.0


def _monthly_mean(values: list[float], months: list[str], target: str) -> float:
    """Mean of `values` whose YYYY-MM equals `target`, or NaN when the month is unobserved."""
    picked = [v for v, m in zip(values, months) if m == target and v == v]
    return sum(picked) / len(picked) if picked else float("nan")


def month_keys(end: str, n: int = NUM_TIMESTEPS) -> list[str]:
    """The `n` YYYY-MM keys ending at (and including) `end`'s month, oldest first."""
    y, m = int(end[:4]), int(end[5:7])
    keys = []
    for back in range(n - 1, -1, -1):
        total = y * 12 + (m - 1) - back
        keys.append(f"{total // 12:04d}-{total % 12 + 1:02d}")
    return keys


def build_input(daily: dict, optical: list[dict], sar: list[dict], elevation: float,
                end_date: str, bands: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One `(12, 17)` normalised input array and its `(12, 17)` mask, ending at `end_date`.

    Mask semantics follow the encoder: **1 means missing**. A channel is masked when its month has
    no observation, so a cloud gap masks that timestep rather than fabricating a value — the same
    rule `radar_block` and `satellite_block` follow.

    `bands` is the raw Sentinel-2 reflectance history from `fetch_bands_history`. Pass it to fill
    the ten optical channels; omit it and they stay masked, which is the reduced-input variant.
    Reflectance is already on Presto's scale (it normalises Earth Engine's 0-10000 by 1e4), so the
    values go in unscaled.
    """
    keys = month_keys(end_date)
    x = np.zeros((NUM_TIMESTEPS, NUM_CHANNELS), dtype=np.float32)
    mask = np.zeros((NUM_TIMESTEPS, NUM_CHANNELS), dtype=np.float32)
    mask[:, OPTICAL_CHANNELS] = 1.0        # unmasked per-month below, only where observed

    band_months = [o["sensing_date"][:7] for o in (bands or [])]
    band_series = {key: [o.get(key, float("nan")) for o in (bands or [])]
                   for key in BAND_CHANNELS.values()}

    daily_months = [t[:7] for t in daily["time"]]
    temps = [(hi + lo) / 2.0 for hi, lo in
             zip(daily["temperature_2m_max"], daily["temperature_2m_min"])]
    rain = daily["precipitation_sum"]

    opt_months = [o["sensing_date"][:7] for o in optical]
    ndvi = [o.get("ndvi", float("nan")) for o in optical]
    sar_months = [o["sensing_date"][:7] for o in sar]
    vv = [o.get("vv", float("nan")) for o in sar]
    vh = [o.get("vh", float("nan")) for o in sar]

    for t, key in enumerate(keys):
        temp_c = _monthly_mean(temps, daily_months, key)
        precip_mm = _monthly_mean(rain, daily_months, key)
        n = _monthly_mean(ndvi, opt_months, key)
        v_vv = _monthly_mean(vv, sar_months, key)
        v_vh = _monthly_mean(vh, sar_months, key)

        # Presto: (Kelvin - 272.15) / 35, and Kelvin = Celsius + 273.15  ->  (Celsius + 1) / 35.
        _set(x, mask, t, IDX_ERA5_TEMP, temp_c, lambda c: (c + 1.0) / ERA5_TEMP_DIV)
        _set(x, mask, t, IDX_ERA5_PRECIP, precip_mm, lambda mm: (mm / 1000.0) / ERA5_PRECIP_DIV)
        _set(x, mask, t, IDX_NDVI, n, lambda v: v)
        _set(x, mask, t, IDX_VV, v_vv, lambda p: (linear_to_db(p) + S1_SHIFT) / S1_DIV)
        _set(x, mask, t, IDX_VH, v_vh, lambda p: (linear_to_db(p) + S1_SHIFT) / S1_DIV)

        for channel, band_key in BAND_CHANNELS.items():
            value = _monthly_mean(band_series[band_key], band_months, key)
            _set(x, mask, t, channel, value, lambda r: r)

    # Static: Presto collapses SRTM to one token, so the value only has to be right at t=0.
    x[:, IDX_ELEVATION] = (elevation or 0.0) / ELEVATION_DIV
    x[:, IDX_SLOPE] = 0.0                  # no slope upstream; 0 == flat after /50
    return x, mask


def _set(x: np.ndarray, mask: np.ndarray, t: int, idx: int, raw: float, transform) -> None:
    """Write a normalised value, or mark the slot missing. Never writes a substitute."""
    if raw == raw:                          # not NaN
        x[t, idx] = transform(raw)
        mask[t, idx] = 0.0
    else:
        mask[t, idx] = 1.0


def load_encoder(weights: Path = WEIGHTS):
    """The frozen pretrained encoder, in eval mode."""
    if not weights.exists():
        raise FileNotFoundError(
            f"{weights} not found. Fetch it once with:\n"
            "  curl -L -o .cache/presto/default_model.pt "
            "https://raw.githubusercontent.com/nasaharvest/presto/main/data/default_model.pt"
        )
    model = Presto.construct()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.encoder


def batch_key(mask: np.ndarray, month: int) -> tuple:
    """Group key for samples that may share an encoder call.

    `Encoder.mask_tokens` asserts every item in a batch has the same number of masked tokens —
    "we assume the number of masked patches is the same for all items in the batch. Otherwise
    things become a headache". Our masks genuinely differ per sample, because a cloud gap in
    March is not a cloud gap in April, so samples have to be bucketed before batching.

    Bucketing on the whole mask pattern rather than just its count: equal counts satisfy the
    assertion, but identical patterns also guarantee the kept/removed token *positions* line up,
    which costs a few more buckets and removes a class of silent misalignment.

    `month` joins the key because the encoder takes one scalar month per call for t=0.
    """
    return (month, mask.tobytes())


def embed(encoder, arrays: list[np.ndarray], masks: list[np.ndarray],
          latlons: list[tuple[float, float]], months: list[int],
          batch_size: int = 256) -> np.ndarray:
    """Frozen 128-dim embeddings for a list of inputs. `months` is the calendar month of t=0."""

    out = np.zeros((len(arrays), 128), dtype=np.float32)
    buckets: dict[tuple, list[int]] = {}
    for i, (mask, month) in enumerate(zip(masks, months)):
        buckets.setdefault(batch_key(mask, month), []).append(i)

    with torch.no_grad():
        for (month, _), idxs in buckets.items():
            for start in range(0, len(idxs), batch_size):
                chunk = idxs[start:start + batch_size]
                x = torch.from_numpy(np.stack([arrays[i] for i in chunk]))
                m_ = torch.from_numpy(np.stack([masks[i] for i in chunk]))
                ll = torch.tensor([latlons[i] for i in chunk], dtype=torch.float)
                dw = torch.full((len(chunk), NUM_TIMESTEPS), DYNAMIC_WORLD_MISSING,
                                dtype=torch.long)
                emb = encoder(x, dynamic_world=dw, latlons=ll, mask=m_, month=month,
                              eval_task=True)
                out[chunk] = emb.numpy()
    return out


def month_of(day: str) -> int:
    """Presto's month index is 0-based; `date.fromisoformat` gives 1-based."""
    return date.fromisoformat(day).month - 1
