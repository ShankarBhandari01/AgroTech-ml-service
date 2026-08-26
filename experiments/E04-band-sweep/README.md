# E04 — the transfer-versus-specificity band sweep

`lat_band x elev_band` swept over `{5, 10, 20, 90} x {500, 1000, 2000}` (12 cells), on `level_z`
(the only `peer_key`-comparable target) and `within_xy` (where E02's positive result lives),
`peer_key=geo_month`, spatial (leave-one-cluster-out) folds only. 24 config files
(`E04-<target>-lat<N>-elev<N>.yaml`) and 24 `.result.json` files sit alongside this README, each
carrying its own provenance block: `git_sha=dd59159`, `dirty=False`, `seed=42`,
`content_hash=6c832dc2741766af8a74b5fd2dd39cab4d4557fdba811ecc728a8a6b8bf28aeb` (same data/commit
as `experiments/E02/`).

**This directory's files are Task F's output. Per the coordinator's later instruction, the working
directory is left uncommitted — nothing here has been `git add`ed or committed.**

## Donor rows (anchor check)

Donor rows = training rows sharing a bucket with *any* of the held-out cluster's own rows. Computed
directly from `argotech.lab.peers.peer_key` and `argotech.lab.splits.leave_one_cluster_out`, no
model fit involved. Order: Benue, Kaduna, Kano, Kenya.

| Band | Benue | Kaduna | Kano | Kenya |
| --- | --- | --- | --- | --- |
| 5°/500 m | 0 | 282 | 1098 | 0 |
| 10°/1000 m | 36 | 1353 | 1098 | 647 |
| 20°/1000 m | 2175 | 2799 | 1792 | 3139 |

All three match the plan's known anchors exactly.

## `level_z`, `boosted` (best arm on this target) — net benefit and per-cluster peer coverage

| lat° | elev m | NB | CI | folds | cov(Benue/Kaduna/Kano/Kenya) |
| --- | --- | --- | --- | --- | --- |
| 5 | 500 | +0.0029 | [-0.0230, +0.0289] | 2/4 | 0.00 / 0.96 / 0.21 / 0.00 |
| 5 | 1000 | +0.0176 | [+0.0068, +0.0284] | 2/4 | 0.00 / 0.96 / 1.00 / 0.00 |
| 5 | 2000 | +0.0141 | [+0.0036, +0.0247] | 2/4 | 0.00 / 1.00 / 1.00 / 0.00 |
| 10 | 500 | +0.0029 | [-0.0230, +0.0289] | 2/4 | 0.00 / 0.96 / 0.21 / 0.00 |
| 10 | 1000 | +0.0132 | [-0.0012, +0.0276] | 4/4 | 1.00 / 0.96 / 1.00 / 0.02 |
| 10 | 2000 | +0.0059 | [-0.0039, +0.0172] | 4/4 | 1.00 / 1.00 / 1.00 / 0.42 |
| 20 | 500 | +0.0077 | [+0.0009, +0.0172] | 4/4 | 1.00 / 1.00 / 0.86 / 0.25 |
| **20** | **1000** | **+0.0066** | **[-0.0020, +0.0157]** | **4/4** | **1.00 / 1.00 / 1.00 / 0.47** |
| 20 | 2000 | +0.0104 | [+0.0037, +0.0209] | 4/4 | 1.00 / 1.00 / 1.00 / 0.55 |
| 90 | 500/1000/2000 | *identical to lat=20, same elev, row for row* | | | |

## `within_xy`, `linear` (the arm carrying E02's +0.0404) and `boosted`

| lat° | elev m | linear NB | linear CI | boosted NB | boosted CI | folds | cov(B/Kd/Kn/Ky) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 500 | +0.0596 | [-0.0096, +0.1289] | +0.0237 | [-0.0077, +0.0550] | 2/4 | 0.00/0.96/0.21/0.00 |
| 5 | 1000 | +0.0152 | [-0.0082, +0.0386] | +0.0132 | [-0.0091, +0.0356] | 2/4 | 0.00/0.96/1.00/0.00 |
| 5 | 2000 | +0.0172 | [-0.0068, +0.0412] | +0.0148 | [-0.0076, +0.0372] | 2/4 | 0.00/1.00/1.00/0.00 |
| 10 | 500 | +0.0596 | [-0.0096, +0.1289] | +0.0237 | [-0.0077, +0.0550] | 2/4 | 0.00/0.96/0.21/0.00 |
| 10 | 1000 | **+0.1580** | [-0.0036, +0.4492] | +0.0722 | [-0.0154, +0.2105] | 4/4 | 1.00/0.96/1.00/0.02 |
| 10 | 2000 | +0.0091 | [-0.0061, +0.0265] | +0.0057 | [-0.0067, +0.0235] | 4/4 | 1.00/1.00/1.00/0.42 |
| 20 | 500 | +0.0567 | [-0.0051, +0.1483] | +0.0296 | [-0.0129, +0.0834] | 4/4 | 1.00/1.00/0.86/0.25 |
| **20** | **1000** | **+0.0404** | **[-0.0072, +0.0940]** | **+0.0218** | **[-0.0051, +0.0542]** | **4/4** | **1.00/1.00/1.00/0.47** |
| 20 | 2000 | +0.0235 | [-0.0173, +0.0808] | +0.0186 | [-0.0050, +0.0513] | 4/4 | 1.00/1.00/1.00/0.55 |
| 90 | 500/1000/2000 | *identical to lat=20, same elev, row for row* | | | | | |

## Finding 1: at 20°, latitude has already stopped discriminating

`lat_band=20` and `lat_band=90` produce **bit-identical** results in every one of the 48
(cell x target x arm) rows checked — same net benefit, same CI, same donor-row counts, same
per-cluster coverage. This is not approximate agreement; the values are equal. The four clusters'
mean latitudes (Benue 7.6°N, Kaduna 10.8°N, Kano 12.0°N, Kenya 0.5°N) all satisfy
`floor(lat/20) == floor(lat/90) == 0`, so every row already lands in the same latitude bucket at
20° that it would at 90°. From 20° up, the key is `elevation-and-month` alone — exactly what
spec §7a's existing text predicts ("at 20 degrees latitude stops discriminating at all ... the key
degrades to elevation-and-month, pooling humid Benue with Sudan-savannah Kano"). This sweep
confirms it numerically rather than by inspection of the band width alone.

## Finding 2: the real constraint at every band is Kenya's coverage, not Benue's or Kano's

At `elev_band=500` (any lat >= 20), Benue, Kaduna and Kano all reach 0.86-1.00 coverage, but Kenya
sits at 0.25 — its 1760 m mean elevation is the outlier (the other three cluster means are
118-676 m; the highlands never land in the same elevation bucket as any lowland cluster until the
band is wide enough to swallow the gap). Coverage for Kenya climbs 0.25 -> 0.47 -> 0.55 as
`elev_band` widens 500 -> 1000 -> 2000, and **net benefit falls as it does**: `within_xy`/`linear`
goes +0.0567 -> +0.0404 -> +0.0235; `boosted` goes +0.0296 -> +0.0218 -> +0.0186; `level_z`/`boosted`
is flatter (+0.0077 -> +0.0066 -> +0.0104) but still never beats the 500 m cell's CI lower bound by
much. Below `elev_band=500` at narrow `lat_band` (5° or 10° with elev=500), Benue's coverage
collapses to 0.00 too (0 donor rows) and only 2/4 folds score at all — both extreme-latitude
clusters (Benue and Kenya) lose their reference simultaneously once the band is too tight in both
dimensions at once.

**Where it turns:** coverage is "complete enough" (3 of 4 clusters >= 0.86, Kenya the only partial
one) starting at `20°/500 m`, and net benefit is highest in this sweep at that same cell
(`within_xy` linear +0.0567). Widening further to `20°/1000 m` (the width this investigation
shipped) buys Kenya coverage +0.22 (0.25 -> 0.47) at a cost of -0.0163 net benefit on `within_xy`
linear (+0.0567 -> +0.0404); `20°/2000 m` buys another +0.08 Kenya coverage for another -0.0169.
Every extra point of Kenya coverage past 500 m costs net benefit somewhere on the curve — that is
the actual transfer/specificity trade, measured rather than assumed.

## Finding 3: the `10°/1000 m` "+0.1580" cell is a degenerate fold, not a sweeter spot

Read off the table, `within_xy`/`linear` peaks at `10°/1000 m` (+0.1580, not +0.0404). Its own
`.result.json` shows why this is not a stronger result — it is a 4-fold mean dominated by one
outlier fold:

| fold | peer_coverage | n | net_benefit | event_rate |
| --- | --- | --- | --- | --- |
| Benue | 1.00 | 566 | **+0.5989** | **0.882** |
| Kaduna | 0.964 | 1005 | +0.0403 | 0.205 |
| Kano | 1.00 | 1257 | -0.0072 | 0.043 |
| Kenya | 0.025 | 43 | 0.0 | 0.070 |

Benue has only 36 donor rows at this band (see the anchor table) — the thinnest non-zero donor
pool of any scored cluster here. A peer reference fit on 36 rows is a noisy, plausibly biased
estimate of Benue's true peer mean/sd; applying it pushes Benue's `event_rate` (share of rows
flagged `ztilde <= tau`) to 88.2%, nearly nine times every other cluster's rate in this same run.
Net benefit under a near-universal event rate is close to its ceiling almost by construction — a
model does not need much skill to look good when the vast majority of held-out rows are positive.
Averaging that fold in with three ordinary ones (event rates 4-21%) produces the reported +0.1580
mean and its very wide CI [-0.0036, +0.4492], which itself crosses zero. **This is reported, not
excluded, but it is not evidence of a better operating point than `20°/1000 m`** — it is evidence
that a thin donor pool (here, 36 rows) can distort the peer reference badly enough to manufacture
an extreme per-fold event rate, and per-fold reporting is what exposes that rather than a fold mean
alone.

## Conclusion

The trade-off is real and it turns between `20°/500 m` (best net benefit in this sweep, but Kenya
only 25% covered) and `20°/1000 m` (the band this investigation shipped: Kenya reaches 47%
coverage, `within_xy` linear net benefit falls from +0.0567 to +0.0404 — matching every anchor already
committed in `experiments/E02/`, but **not** "clearly positive": that cell's CI is
[-0.0072, +0.0940] and crosses zero, as does every other cell's). Beyond `20°/1000 m`, widening
further keeps buying Kenya coverage but at continuing net-benefit cost, and beyond `20°` latitude
stops doing anything at all (Finding 1). Narrower than `20°` collapses coverage for one or both
extreme-latitude clusters outright (Finding 2), and at least one narrow cell's apparently-strong
number is a donor-pool artifact rather than a transfer improvement (Finding 3). `20°/1000 m` is a
defensible, if not literally optimal-by-net-benefit, choice: it is the point past which coverage
gains keep costing net benefit indefinitely, rather than the point of maximum net benefit outright.

## The caveat that governs every number above

**None of the 12 `within_xy` cells has a bootstrap CI excluding zero — 0 of 12.** The same holds
across the `level_z` sweep. So the ordering discussed above is an ordering of point estimates whose
intervals all overlap zero and each other, on four leave-one-cluster-out folds.

Read literally, this sweep measures a **coverage** curve, not a skill curve. Coverage responds
strongly and predictably to band width (0.29 to 0.89 mean, and the per-cluster columns show exactly
which cluster each widening buys). Net benefit does not separate from zero at any band width.

`20°/1000 m` is therefore defensible as the band that maximises coverage before latitude stops
discriminating — a coverage argument, which the data supports. It is **not** defensible as the band
that maximises transferable skill, because no band in this sweep demonstrates transferable skill at
all. Two confounds make the net-benefit column unreadable as a skill-versus-band curve in any case:
low-coverage cells score only 2 of 4 folds (so those are two-cluster LOCO numbers), and the event
rate varies 0.124 to 0.300 across cells while the maximum attainable net benefit *is* the event
rate, so cells are not on a common scale.
