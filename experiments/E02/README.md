# E02 matrix — the full 12-cell run, under the provenance rule

Every cell below is a config file (`E02-<target>-<peer_key>.yaml`) and a `.result.json` in this
directory, each carrying a `provenance` block: data manifest hash, git SHA, dirty flag, seed. The
numbers in this README are not asserted — they are transcribed from those files, all produced by
the same run:

    git_sha=dd59159  dirty=False  seed=42  content_hash=6c832dc2741766af8a74b5fd2dd39cab4d4557fdba811ecc728a8a6b8bf28aeb

(`level_z`/`within_*` cells score 4230 rows; `delta_z` cells score 4596 — `delta_z` needs no
`alpha_hat` history floor the other three kinds require, per `targets.build_target`.)

Base config: `experiments/E02-within-vs-level.yaml`. Each of the 12 files here changes only `name`,
`target` and `peer_key` from that base — same features, arms, splits, `min_history`, `shrink`,
`tau=-1.0`, `seed=42`, `lat_band=20.0`, `elev_band=1000.0`.

## The cross-key caveat (verbatim)

> Only `level_z` supports comparison across `peer_key` values. `within_y`, `within_xy` and `delta_z`
> targets are defined *through* `ndvi_z_peer`, so changing the peer key changes the target itself.
> `run.py` warns at runtime when that combination runs.

Everything below that is not `level_z` is grouped by `peer_key` for legibility, not because it is
comparable across `peer_key`. A `within_xy`/`geo_month` number and a `within_xy`/`leaky` number are
scores on two different targets that happen to share a name.

## Every spatial `cluster_month` cell skipped — 4/4 folds, all four targets

`level_z`, `within_y`, `within_xy` and `delta_z` under `peer_key=cluster_month`, split=`spatial`,
all report:

    [spatial]  0/4 folds scored — every fold was skipped

A held-out cluster's `cluster_month` bucket is `<cluster>|MM`, keyed on the cluster's own identity —
so a cluster that is entirely held out (leave-one-cluster-out) never appears in its own training
fit and gets no peer bucket at all. With no `ndvi_z_peer` reference, `within_y`/`within_xy`/`delta_z`
cannot even build their target (they are defined through `ndvi_z_peer`), and `level_z`'s
`build_target` still drops every row with no `alpha_hat`/reference history in the same fold. This is
the correct, honest behaviour of a fold-fitted reference under an honest cluster key — **the
finding, not a failure of the run.** (`cluster_month`'s *temporal* folds are unaffected: temporal
folds hold out a time window, not a cluster identity, so every cluster is present in every temporal
training fit.)

## The one valid cross-key table: `level_z`, spatial and temporal, `boosted` (best arm)

| peer_key | spatial NB | spatial CI | temporal NB | temporal CI |
| --- | --- | --- | --- | --- |
| leaky | +0.0064 | [-0.0014, +0.0142] | +0.0194 | [+0.0182, +0.0216] |
| cluster_month | *(all folds skipped)* | — | +0.0179 | [+0.0152, +0.0202] |
| geo_month | +0.0066 | [-0.0020, +0.0157] | +0.0172 | [+0.0162, +0.0182] |

Matches the anchors carried forward from the plan exactly: spatial leaky +0.0064 / geo_month
+0.0066; temporal leaky +0.0194 / cluster_month +0.0179 / geo_month +0.0172.

Every `level_z` cell's full arm ranking (`zero`, `persistence`, `climatology`, `linear`, `boosted`)
is in its own `.result.json`; `boosted` wins net benefit in every scored split here, `climatology`
converges to `zero` on spatial folds (the field effect `level_z` still carries has nothing for a
per-field climatology mean to separate from a fold-invariant zero once no learned signal beats it).

## Best result anywhere: `within_xy` / `geo_month` / spatial / `linear`

| arm | net benefit | CI | folds |
| --- | --- | --- | --- |
| linear | **+0.0404** | [-0.0072, +0.0940] | 4/4 |
| persistence | +0.0317 | [-0.0948, +0.1582] | 4/4 |
| boosted | +0.0218 | [-0.0051, +0.0542] | 4/4 |
| zero | +0.0000 | [+0.0000, +0.0000] | 4/4 |
| climatology | +0.0000 | [+0.0000, +0.0000] | 4/4 |

Matches the anchor exactly: linear +0.0404, boosted +0.0218, zero +0.0000. This is the strongest
result in the matrix and it comes from a target (`within_xy`) that is only comparable to other
`geo_month`-keyed `within_xy` runs, not to `within_xy`/`leaky` or `within_xy`/`cluster_month` above
it in this file — see the cross-key caveat. It rests on 4 folds; Task E (`experiments/E03-replication/`)
replicates it.

## Full matrix (informational — every non-`level_z` row is single-key, not cross-key comparable)

Net benefit (`_mean`), best-net-benefit arm named, spatial then temporal. "SKIP" = every fold in
that split was skipped.

| target | peer_key | spatial best arm (NB) | temporal best arm (NB) |
| --- | --- | --- | --- |
| level_z | leaky | boosted (+0.0064) | boosted (+0.0194) |
| level_z | cluster_month | SKIP | boosted (+0.0179) |
| level_z | geo_month | boosted (+0.0066) | boosted (+0.0172) |
| within_y | leaky | zero/climatology (+0.0000) | zero (+0.0000) |
| within_y | cluster_month | SKIP | zero (+0.0000) |
| within_y | geo_month | persistence (+0.0317) | boosted (+0.0023) |
| within_xy | leaky | zero/climatology (+0.0000) | zero (+0.0000) |
| within_xy | cluster_month | SKIP | zero (+0.0000) |
| within_xy | geo_month | linear (+0.0404) | zero (+0.0000) |
| delta_z | leaky | boosted (+0.0155) | boosted (+0.0172) |
| delta_z | cluster_month | SKIP | linear (+0.0319) |
| delta_z | geo_month | linear (+0.1530) | boosted (+0.0731) |

`delta_z`'s `geo_month` numbers are large and wide (`spatial` CI [-0.0182, +0.3326], 4 folds, tiny
per-fold n after the `alpha_hat`-free row count) — reported here for completeness, not endorsed as a
finding; they are exactly the kind of number this repo's provenance rule exists to let a reader
re-derive and interrogate rather than cite blind. `delta_z` is out of scope for Task E/F (only
`within_xy` and `level_z` are pursued further, per the plan).
