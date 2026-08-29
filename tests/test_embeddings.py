"""Checks on the Presto input encoding.

Every assertion here guards a *silent* failure. Wrong units, a shifted channel index or an inverted
mask all produce a perfectly healthy-looking 128-dim embedding that means nothing, and the only
symptom downstream is a model that mysteriously fails to beat its baseline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from argotech.lab.presto.embeddings import (
    IDX_ELEVATION,
    IDX_ERA5_PRECIP,
    IDX_ERA5_TEMP,
    IDX_NDVI,
    IDX_VH,
    IDX_VV,
    NUM_CHANNELS,
    NUM_TIMESTEPS,
    OPTICAL_CHANNELS,
    build_input,
    linear_to_db,
    month_keys,
    month_of,
)


def _daily(months: list[str], temp_c: float = 24.0, rain_mm: float = 3.0) -> dict:
    times, tmax, tmin, rain = [], [], [], []
    for m in months:
        for day in range(1, 29):
            times.append(f"{m}-{day:02d}")
            tmax.append(temp_c + 6.0)
            tmin.append(temp_c - 6.0)
            rain.append(rain_mm)
    return {"time": times, "temperature_2m_max": tmax, "temperature_2m_min": tmin,
            "precipitation_sum": rain}


def test_month_keys_walks_back_twelve_months_across_a_year_boundary():
    keys = month_keys("2025-02-14")
    assert len(keys) == NUM_TIMESTEPS
    assert keys[-1] == "2025-02", "last timestep is the prediction month"
    assert keys[0] == "2024-03", "twelve months back, inclusive"
    assert keys == sorted(keys), "oldest first"


def test_linear_to_db_matches_the_definition_and_survives_zero():
    assert linear_to_db(1.0) == 0.0
    assert abs(linear_to_db(0.1) - (-10.0)) < 1e-9
    # CDSE returns 0 for a fully masked AOI; log10(0) would be -inf and poison the encoder.
    assert linear_to_db(0.0) == -50.0
    assert math.isfinite(linear_to_db(0.0))


def test_sentinel1_is_converted_to_decibels_before_normalising():
    """The conversion that fails silently: Presto trained on dB, our evalscript returns linear.

    Linear 0.1 is -10 dB, which normalises to (-10 + 25) / 25 = 0.6. Feeding the linear value
    unconverted would give (0.1 + 25) / 25 = 1.004 — plausible-looking and wrong.
    """
    months = month_keys("2025-06-15")
    sar = [{"sensing_date": f"{m}-15", "vv": 0.1, "vh": 0.01, "rvi": 0.5} for m in months]
    x, mask = build_input(_daily(months), [], sar, elevation=500.0, end_date="2025-06-15")

    assert abs(x[-1, IDX_VV] - 0.6) < 1e-5, "VV: 0.1 linear -> -10 dB -> (-10+25)/25 = 0.6"
    assert abs(x[-1, IDX_VH] - 0.2) < 1e-5, "VH: 0.01 linear -> -20 dB -> (-20+25)/25 = 0.2"
    assert (mask[:, IDX_VV] == 0).all(), "observed radar must not be masked"
    # Guard against the un-converted value sneaking through.
    assert x[-1, IDX_VV] != pytest.approx((0.1 + 25.0) / 25.0)


def test_precipitation_is_converted_from_millimetres_to_metres():
    """Earth Engine ERA5 total_precipitation is metres/day; Open-Meteo gives millimetres/day."""
    months = month_keys("2025-06-15")
    x, _ = build_input(_daily(months, rain_mm=30.0), [], [], elevation=0.0, end_date="2025-06-15")
    # 30 mm -> 0.03 m -> /0.03 -> 1.0
    assert abs(x[-1, IDX_ERA5_PRECIP] - 1.0) < 1e-5
    # The factor-of-1000 error would give 1000.0.
    assert x[-1, IDX_ERA5_PRECIP] < 2.0


def test_temperature_normalisation_matches_the_kelvin_formula():
    """Presto: (Kelvin - 272.15) / 35. Ours is Celsius, so (C + 1) / 35."""
    months = month_keys("2025-06-15")
    x, _ = build_input(_daily(months, temp_c=24.0), [], [], elevation=0.0, end_date="2025-06-15")
    expected = ((24.0 + 273.15) - 272.15) / 35.0
    assert abs(x[-1, IDX_ERA5_TEMP] - expected) < 1e-5


def test_optical_reflectance_channels_are_masked_everywhere():
    """We store computed indices, not band means, so every reflectance slot is withheld."""
    months = month_keys("2025-06-15")
    _, mask = build_input(_daily(months), [], [], elevation=0.0, end_date="2025-06-15")
    assert (mask[:, OPTICAL_CHANNELS] == 1.0).all()
    assert len(OPTICAL_CHANNELS) == 10


def test_an_unobserved_month_is_masked_rather_than_filled():
    """A cloud gap must mask the timestep, never substitute a value."""
    months = month_keys("2025-06-15")
    optical = [{"sensing_date": f"{m}-15", "ndvi": 0.6} for m in months[:6]]  # last 6 missing
    x, mask = build_input(_daily(months), optical, [], elevation=0.0, end_date="2025-06-15")

    assert (mask[:6, IDX_NDVI] == 0).all(), "observed months unmasked"
    assert (mask[6:, IDX_NDVI] == 1).all(), "unobserved months masked"
    assert (x[6:, IDX_NDVI] == 0).all(), "masked slots hold no fabricated value"
    assert np.isfinite(x).all(), "never NaN or inf: the encoder would propagate it"


def test_elevation_is_normalised_and_shape_is_the_encoder_contract():
    months = month_keys("2025-06-15")
    x, mask = build_input(_daily(months), [], [], elevation=1000.0, end_date="2025-06-15")
    assert x.shape == (NUM_TIMESTEPS, NUM_CHANNELS) == mask.shape
    assert abs(x[0, IDX_ELEVATION] - 0.5) < 1e-9, "1000 m / 2000"
    assert x.dtype == np.float32


def test_month_of_is_zero_based():
    """Presto's month embedding indexes 0-11; date.month is 1-12."""
    assert month_of("2025-01-15") == 0
    assert month_of("2025-12-15") == 11


def test_batches_are_grouped_so_every_item_shares_a_mask_pattern():
    """`Encoder.mask_tokens` asserts equal masked-token counts across a batch.

    Real samples violate that — a cloud gap in March is not a cloud gap in April — so bucketing is
    required, not an optimisation.
    """
    from argotech.lab.presto.embeddings import batch_key

    a = np.zeros((NUM_TIMESTEPS, NUM_CHANNELS), dtype=np.float32)
    b = a.copy()
    b[3, IDX_NDVI] = 1.0

    assert batch_key(a, 5) == batch_key(a.copy(), 5)
    assert batch_key(a, 5) != batch_key(b, 5), "different gaps must not share a batch"
    assert batch_key(a, 5) != batch_key(a, 6), "different months must not share a batch"


def test_end_to_end_embedding_is_finite_and_varies_with_input():
    """The real encoder on real-shaped input. Guards the whole pipeline, not just the arithmetic."""
    pytest.importorskip("torch")
    from pathlib import Path

    from argotech.lab.presto.embeddings import embed, load_encoder

    if not Path(".cache/presto/default_model.pt").exists():
        pytest.skip("Presto weights not fetched")

    encoder = load_encoder()
    months = month_keys("2025-06-15")
    wet = _daily(months, temp_c=22.0, rain_mm=12.0)
    dry = _daily(months, temp_c=34.0, rain_mm=0.0)
    sar = [{"sensing_date": f"{m}-15", "vv": 0.12, "vh": 0.03, "rvi": 0.7} for m in months]
    opt = [{"sensing_date": f"{m}-15", "ndvi": 0.7} for m in months]

    x1, m1 = build_input(wet, opt, sar, 500.0, "2025-06-15")
    x2, m2 = build_input(dry, opt, sar, 500.0, "2025-06-15")
    out = embed(encoder, [x1, x2], [m1, m2], [(10.8, 7.9), (10.8, 7.9)], [5, 5])

    assert out.shape == (2, 128)
    assert np.isfinite(out).all(), "a NaN here means an unmasked missing value slipped through"
    assert not np.allclose(out[0], out[1]), "a wet and a dry year must not embed identically"


def test_bands_fill_the_optical_channels_in_prestos_order():
    """A transposed band is invisible: every slot is a valid reflectance value.

    Each band gets a distinct value so a swap between, say, B8 and B8A shows up as a mismatch
    rather than as a plausible-looking number in the wrong place.
    """
    from argotech.lab.presto.embeddings import BAND_CHANNELS

    months = month_keys("2025-06-15")
    probe = {"b02": 0.02, "b03": 0.03, "b04": 0.04, "b05": 0.05, "b06": 0.06,
             "b07": 0.07, "b08": 0.08, "b8a": 0.09, "b11": 0.11, "b12": 0.12}
    bands = [{"sensing_date": f"{m}-15", **probe} for m in months]
    x, mask = build_input(_daily(months), [], [], 0.0, "2025-06-15", bands=bands)

    for channel, key in BAND_CHANNELS.items():
        assert abs(x[-1, channel] - probe[key]) < 1e-6, f"channel {channel} should hold {key}"
        assert mask[-1, channel] == 0.0, f"observed band {key} must be unmasked"

    # Presto's ordering, spelled out: B8A sits at 9, between B8 (8) and B11 (10).
    assert BAND_CHANNELS[8] == "b08" and BAND_CHANNELS[9] == "b8a" and BAND_CHANNELS[10] == "b11"
    assert sorted(BAND_CHANNELS) == OPTICAL_CHANNELS


def test_bands_are_unscaled_because_cdse_already_matches_prestos_scale():
    """Presto divides Earth Engine's 0-10000 by 1e4; CDSE gives float reflectance already."""
    months = month_keys("2025-06-15")
    bands = [{"sensing_date": f"{m}-15", **{k: 0.25 for k in
              ("b02","b03","b04","b05","b06","b07","b08","b8a","b11","b12")}} for m in months]
    x, _ = build_input(_daily(months), [], [], 0.0, "2025-06-15", bands=bands)
    assert abs(x[-1, 2] - 0.25) < 1e-6, "no extra /1e4 and no *1e4"


def test_omitting_bands_keeps_the_reduced_input_variant_working():
    """Both variants must stay runnable so the comparison is controlled."""
    months = month_keys("2025-06-15")
    _, mask = build_input(_daily(months), [], [], 0.0, "2025-06-15", bands=None)
    assert (mask[:, OPTICAL_CHANNELS] == 1.0).all()


def test_a_cloudy_month_masks_only_that_timestep_of_the_bands():
    months = month_keys("2025-06-15")
    probe = {k: 0.2 for k in ("b02","b03","b04","b05","b06","b07","b08","b8a","b11","b12")}
    bands = [{"sensing_date": f"{m}-15", **probe} for m in months[:8]]   # last 4 cloudy
    _, mask = build_input(_daily(months), [], [], 0.0, "2025-06-15", bands=bands)
    assert (mask[:8, OPTICAL_CHANNELS] == 0.0).all()
    assert (mask[8:, OPTICAL_CHANNELS] == 1.0).all()


def test_a_failed_fetch_is_never_cached():
    """The defect that poisoned 42% of a band fetch.

    `_stats` returns [] for both "no imagery" and "request failed", so caching the empty list
    memoises a transient CDSE 429 as a permanent fact about the site.
    """
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    from argotech.lab.panel.panel import _cached_fetch

    with tempfile.TemporaryDirectory() as d:
        path = _Path(d) / "site.json"

        assert _cached_fetch(path, list) == []
        assert not path.exists(), "an empty (possibly failed) result must not be cached"

        good = [{"sensing_date": "2025-01-31", "b02": 0.1}]
        assert _cached_fetch(path, lambda: good) == good
        assert path.exists() and _json.loads(path.read_text()) == good

        # Second call is served from cache without re-fetching.
        assert _cached_fetch(path, lambda: (_ for _ in ()).throw(AssertionError("refetched"))) == good
