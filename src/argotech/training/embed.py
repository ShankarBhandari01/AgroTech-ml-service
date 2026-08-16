"""Precompute frozen Presto embeddings for every sample in the training set.

Separate from `dataset.py` because it is a different kind of expensive: the dataset builder is
network-bound against CDSE and Open-Meteo, this is CPU-bound against a 402K-parameter transformer.
Keeping them apart means a modelling experiment can re-embed without re-fetching, which is the whole
reason the upstream caches exist.

Output is keyed on `(site_id, obs_date)` so `train.py` can merge it onto the sample frame without
either file needing to know the other's row order.

Run: `python -m argotech.training.embed --data data/training_set_sar.parquet`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from argotech.models import embeddings as emb
from argotech.training.dataset import bands_history, collect_site, sample_sites

EMBED_DIM = 128
EMBED_PREFIX = "presto_"


def build_rows(sites_per_cluster: int, years: int, wanted: set[tuple[str, str]],
               with_bands: bool = True) -> pd.DataFrame:
    """One embedding row per `(site_id, obs_date)` present in `wanted`.

    Re-collects the sites, which is cheap: every upstream call is served from `.cache/`. Sites that
    fail to collect are skipped rather than fatal — the merge in `train.py` is a left join, so a
    missing embedding degrades that sample to the tabular features alone.
    """
    encoder = emb.load_encoder()
    have_samples = {site_id for site_id, _ in wanted}

    arrays, masks, latlons, months, keys, missing_bands = [], [], [], [], [], []
    for site in sample_sites(sites_per_cluster):
        # Skip sites that produced no samples. `collect_site` would otherwise re-attempt their
        # upstream fetches, and a site fails precisely because those fetches fail — so each one
        # costs five retries with exponential backoff (~30s) to rediscover what the parquet
        # already tells us.
        if site["site_id"] not in have_samples:
            continue
        collected = collect_site(site, years)
        if not collected:
            continue
        # Fetched here rather than in `collect_site` so a tabular dataset rebuild never pays for it.
        bands = bands_history(site, years * 365) if with_bands else None
        if with_bands and not bands:
            missing_bands.append(site["site_id"])
        for obs in collected["history"]:
            key = (site["site_id"], obs["sensing_date"])
            if key not in wanted:
                continue
            x, mask = emb.build_input(
                daily=collected["daily"],
                optical=collected["history"],
                sar=collected.get("sar", []),
                elevation=collected.get("elevation", 0.0),
                end_date=obs["sensing_date"],
                bands=bands,
            )
            arrays.append(x)
            masks.append(mask)
            latlons.append((site["latitude"], site["longitude"]))
            months.append(emb.month_of(obs["sensing_date"]))
            keys.append(key)

    if not arrays:
        return pd.DataFrame(columns=["site_id", "obs_date"])

    if missing_bands:
        # Loud and fatal-by-default. Band coverage that fails in *whole clusters* — which is what a
        # rate-limit burst produces — confounds any leave-one-cluster-out comparison with cluster
        # identity, and "100% of pairs embedded" hides it completely: every sample still gets a
        # vector, just one with the optical channels masked.
        by_cluster: dict[str, int] = {}
        for site_id in missing_bands:
            by_cluster[site_id.rsplit("-", 1)[0]] = by_cluster.get(site_id.rsplit("-", 1)[0], 0) + 1
        raise RuntimeError(
            f"{len(missing_bands)} sites have no Sentinel-2 bands, by cluster: {by_cluster}. "
            "Re-run to fetch them (failures are no longer cached), or pass --no-bands for the "
            "reduced-input variant. Refusing to emit embeddings with cluster-structured gaps."
        )

    print(f"embedding {len(arrays)} samples ...", flush=True)
    vectors = emb.embed(encoder, arrays, masks, latlons, months)

    out = pd.DataFrame(vectors, columns=[f"{EMBED_PREFIX}{i}" for i in range(EMBED_DIM)])
    out["site_id"] = [k[0] for k in keys]
    out["obs_date"] = [k[1] for k in keys]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training_set_sar.parquet")
    ap.add_argument("--out", default="data/presto_embeddings.parquet")
    ap.add_argument("--sites", type=int, default=32)
    ap.add_argument("--years", type=int, default=4)
    ap.add_argument("--no-bands", action="store_true",
                    help="mask the ten optical reflectance channels (the reduced-input variant)")
    args = ap.parse_args()

    samples = pd.read_parquet(args.data, columns=["site_id", "obs_date"])
    wanted = set(zip(samples.site_id, samples.obs_date))
    print(f"{len(wanted)} (site, date) pairs to embed")

    out = build_rows(args.sites, args.years, wanted, with_bands=not args.no_bands)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    covered = len(out)
    print(f"wrote {args.out}: {covered}/{len(wanted)} pairs "
          f"({covered / max(1, len(wanted)) * 100:.1f}% coverage), {EMBED_DIM} dims")
    if covered:
        vals = out[[c for c in out.columns if c.startswith(EMBED_PREFIX)]].to_numpy()
        print(f"embedding stats: mean {vals.mean():+.4f}  sd {vals.std():.4f}  "
              f"finite {np.isfinite(vals).all()}")


if __name__ == "__main__":
    main()
