# Split serving from the lab — two install surfaces, one repository

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Make it impossible for research code to reach the production image, without splitting the
shared domain layer that keeps training and serving honest.

**Why this is cheap.** The dependency graph is already one-directional and clean, measured:
`serving` imports 24 `argotech.*` modules, all of them `domain`, `data`, `features`,
`models.registry`, `config`, `serving`. `lab` imports the same shared core. **Neither imports the
other.** So this is packaging and enforcement, not refactoring.

**What ships today that should not.** `Dockerfile` does `COPY src src`, so the production image
contains `argotech/lab/` (220K, 11 modules) and `models/_presto_vendored.py` (28K, 766 lines of
torch code) — **2,475 lines** that are never imported at serving time.

## Global Constraints

- Python >=3.11. `venv/bin/python3 -m pytest` (or `venv/bin/pytest`, now fixed). Baseline: 179 passed, 1 skipped.
- **`[project.dependencies]` must not grow.** The serving image is the thing being protected.
- Ruff clean. Reproduction gate must still print `0.345 [0.2357, 0.4353] 0.0`.
- **A panel rebuild may be running** — do not touch `.cache/` or `data/`.

---

### Task 1: move the research-only modules into `lab/`

`models/` is currently split across both surfaces, which makes a clean packaging boundary
impossible. `models/registry.py` is serving's (it loads the artifact); `embeddings.py` and
`_presto_vendored.py` are research-only and drag in torch.

- [ ] `git mv src/argotech/models/embeddings.py src/argotech/lab/embeddings.py`
- [ ] `git mv src/argotech/models/_presto_vendored.py src/argotech/lab/_presto_vendored.py`
- [ ] Repoint every import. `tests/test_embeddings.py` and `lab/embed.py` are the known consumers — grep for others rather than assuming.
- [ ] Update `pyproject.toml`'s `extend-exclude` for the vendored file's new path (it is kept byte-comparable to the upstream release so it can be re-vendored by diff; reformatting would destroy that).
- [ ] After this, `models/` contains only `registry.py` and is entirely core.
- [ ] Full suite, ruff, commit.

### Task 2: the boundary test — this is what actually enforces it

Packaging stops research code reaching the *image*. A test stops it reaching the *codebase*, and
fails locally and in CI rather than in production.

- [ ] Create `tests/test_import_boundary.py`:

```python
"""serving must never import lab. The packaging split makes this true in the image; this test
makes it true in the repository, and fails here rather than in a container.

The dependency runs one way by design: lab and serving share domain/, data/ and features/, and
neither imports the other. That property is what lets the production image exclude lab entirely.
"""

from __future__ import annotations

import importlib
import sys


def _argotech_modules_after_importing(target: str) -> set[str]:
    for mod in [m for m in sys.modules if m.startswith("argotech")]:
        del sys.modules[mod]
    importlib.import_module(target)
    return {m for m in sys.modules if m.startswith("argotech")}


def test_serving_does_not_import_lab():
    used = _argotech_modules_after_importing("argotech.serving.main")
    lab = {m for m in used if m.startswith("argotech.lab")}
    assert not lab, f"serving reached into the lab: {sorted(lab)}"


def test_the_nightly_job_does_not_import_lab():
    """`jobs.precompute` runs in the production image too, so it is bound by the same rule."""
    used = _argotech_modules_after_importing("argotech.jobs.precompute")
    lab = {m for m in used if m.startswith("argotech.lab")}
    assert not lab, f"the nightly job reached into the lab: {sorted(lab)}"


def test_lab_may_import_the_shared_core():
    """The reverse direction is allowed and load-bearing: one feature builder, used by both paths,
    is what stops a feature meaning two different things at train and serve time."""
    used = _argotech_modules_after_importing("argotech.lab.run")
    assert "argotech.features.agronomic" in used
    assert not {m for m in used if m.startswith("argotech.serving")}, "lab must not import serving"
```

- [ ] Verify it bites: temporarily add `from argotech.lab import arms` to `serving/main.py`, watch the test fail, remove it. Report both outputs.
- [ ] Commit.

### Task 3: two install surfaces

- [ ] In `pyproject.toml`, exclude the lab package from the default (serving) install:

```toml
[tool.setuptools.packages.find]
where = ["src"]
exclude = ["argotech.lab*"]
```

- [ ] Rename the `train` extra to `lab` (keeping `train` as an alias if anything references it — grep CI and docs first), and confirm it carries `pyarrow`, `torch`, `einops`, `pyyaml`, `mlflow`.
- [ ] Document in `pyproject.toml`, in a comment, that the default install is the **serving surface** and `.[lab]` plus a source checkout is the **research surface**. State the reason: the production image must not contain research code, and the exclusion is what guarantees it rather than the Dockerfile remembering to.
- [ ] Verify a clean install of the default target does NOT provide `argotech.lab`, and that an editable install from the repo still gives the lab to developers.
- [ ] Full suite, ruff, commit.

### Task 4: the Dockerfile stops copying the lab

- [ ] Replace `COPY src src` with copies of only the serving surface, OR keep it and rely on the
      packages-find exclusion — **whichever you choose, prove it**: build the image and assert
      `python -c "import argotech.lab"` fails inside it while `import argotech.serving.main`
      succeeds. Include the output in your report.
- [ ] Confirm the image still starts and answers `/health`.
- [ ] Commit.

### Task 5: write down where things live

- [ ] Add a short section to `README.md` stating the two surfaces and the rule:

| | contents | installed by |
| --- | --- | --- |
| **serving surface** (default) | `config`, `data`, `domain`, `features`, `models.registry`, `jobs`, `serving` | `pip install .` — what the image gets |
| **research surface** | the above **plus** `lab` (panel, targets, peers, splits, arms, evaluate, run, export, embed, presto) | `pip install -e '.[lab]'` from a checkout |

- [ ] And where the non-code artifacts live, with the reason for each:
  - `experiments/` — **committed**. Results are the record; each carries a manifest hash, git SHA and seed.
  - `notebooks/` — **committed, never packaged.** Reads committed CSVs and the panel.
  - `data/` — **gitignored**, one parquet whitelisted. A derived CSV in git is a second source of truth that drifts.
  - `artifacts/` — the model bundle serving loads; produced only by `argotech.lab.export`.
- [ ] Commit.

## Deliberately NOT in this plan

- Two repositories. It would split `domain/`, and a duplicated domain layer is how train/serve skew
  returns — this project has already been bitten by exactly that (`docs/model-design.md` P0-2).
- DVC or object storage. The panel is 479KB and rebuilt from cached upstreams; the content-hash
  manifest already provides the traceability half without the machinery.
- Publishing either surface to an index. Nothing needs it yet.
