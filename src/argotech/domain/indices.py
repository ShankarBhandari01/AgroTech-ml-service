"""Spectral indices and the Crop Health Index.

Pure functions over already-fetched reflectance / index values. No I/O, no model, no labels —
everything here is computable on day one for any field on Earth, which is why it is the layer the
product should lean on while supervised labels are still being collected.

Band references are Sentinel-2 L2A: B02 blue, B04 red, B05/B06/B07 red-edge, B08 NIR, B11 SWIR1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict


def _norm_diff(a: float, b: float) -> float:
    denom = a + b
    return 0.0 if denom == 0 else (a - b) / denom


def ndvi(nir: float, red: float) -> float:
    """Greenness / biomass density. Saturates above LAI ~3 — see `evi` and `ndre`."""
    return _norm_diff(nir, red)


def ndmi(nir: float, swir1: float) -> float:
    """Canopy water content (B08/B11). The literature name for what this repo calls "NDWI"."""
    return _norm_diff(nir, swir1)


def ndre(nir: float, red_edge: float) -> float:
    """Red-edge index (B08/B05). Tracks canopy nitrogen and mid-season stress after NDVI saturates."""
    return _norm_diff(nir, red_edge)


def evi(nir: float, red: float, blue: float) -> float:
    """Soil/aerosol-corrected greenness. Preferred over NDVI on dense canopy."""
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    return 0.0 if denom == 0 else 2.5 * (nir - red) / denom


def savi(nir: float, red: float, soil_factor: float = 0.5) -> float:
    """Soil-Adjusted Vegetation Index. Use in place of NDVI on the sparse canopy typical of
    semi-arid smallholder plots early in the season, where bare soil dominates the pixel."""
    denom = nir + red + soil_factor
    return 0.0 if denom == 0 else (nir - red) * (1.0 + soil_factor) / denom


def anomaly_z(value: float, baseline_mean: float, baseline_std: float) -> float:
    """Standardised anomaly against a per-field or per-peer-group baseline.

    This is the single most important transform in the whole system: an absolute NDVI of 0.45 is
    excellent for sorghum in the Sahel and alarming for irrigated maize, so raw indices must never
    be fed to a model or a threshold without a reference distribution.
    """
    if baseline_std <= 1e-6:
        return 0.0
    return (value - baseline_mean) / baseline_std


def vci(current_ndvi: float, ndvi_min: float, ndvi_max: float) -> float:
    """Vegetation Condition Index (0-100). Current greenness placed within the field's own
    historical range for the same point in the season. The standard drought-monitoring transform
    (Kogan 1990); < 35 is conventionally read as drought stress."""
    span = ndvi_max - ndvi_min
    if span <= 1e-6:
        return 50.0
    return _clamp01((current_ndvi - ndvi_min) / span) * 100.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _logistic(x: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


@dataclass
class CropHealth:
    """Explainable 0-100 health score plus the sub-scores it was composed from."""

    score: float
    vigour: float       # relative greenness within the field's own history (VCI)
    moisture: float     # canopy water content
    anomaly: float      # how this field compares to its peer group right now
    status: str

    def as_dict(self) -> dict:
        return asdict(self)


def crop_health_index(
    current_ndvi: float,
    ndmi_value: float,
    ndvi_min: float,
    ndvi_max: float,
    peer_mean: float,
    peer_std: float,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> CropHealth:
    """Composite Crop Health Index on 0-100.

    Deliberately a transparent weighted mean of three normalised sub-scores rather than a learned
    model: it needs no labels, it is stable across seasons, and an agronomist can audit why a field
    scored what it scored. It is a *state* measure — how the canopy looks now — and must not be
    confused with the forward-looking risk score in `domain.risk`.
    """
    vigour = vci(current_ndvi, ndvi_min, ndvi_max) / 100.0
    # NDMI in the field-relevant range [-0.2, 0.4]; below -0.2 the canopy is effectively bare.
    moisture = _clamp01((ndmi_value + 0.2) / 0.6)
    # Peer anomaly: 1 sigma below the peer group maps to ~0.27, 1 sigma above to ~0.73.
    anomaly = _logistic(anomaly_z(current_ndvi, peer_mean, peer_std), steepness=1.0)

    w_v, w_m, w_a = weights
    total = w_v + w_m + w_a
    score = 100.0 * (w_v * vigour + w_m * moisture + w_a * anomaly) / total

    if score >= 70:
        status = "Healthy"
    elif score >= 45:
        status = "Marginal"
    elif score >= 25:
        status = "Stressed"
    else:
        status = "Severely Stressed"

    return CropHealth(
        score=round(score, 1),
        vigour=round(vigour, 3),
        moisture=round(moisture, 3),
        anomaly=round(anomaly, 3),
        status=status,
    )
