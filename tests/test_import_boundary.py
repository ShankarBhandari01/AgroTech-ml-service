"""serving must never import lab. The packaging split makes this true in the image; this test
makes it true in the repository, and fails here rather than in a container.

The dependency runs one way by design: lab and serving share domain/, data/ and features/, and
neither imports the other. That property is what lets the production image exclude lab entirely.
"""

from __future__ import annotations

import importlib
import sys


def _argotech_modules_after_importing(target: str) -> set[str]:
    """What `argotech.*` modules importing `target` pulls in, judged from a clean slate.

    The clean slate is local to this call: the `argotech.*` corner of `sys.modules` is snapshotted
    first and restored after, so this test's own forced re-imports never leak into whichever test
    runs next. Without the restore, a later test holds a `store`/`pipeline`/`db` reference captured
    before this ran while the *next* thing that does `import argotech.data.store` gets the fresh
    copy this function leaves behind — two live copies of a module that is supposed to be a
    singleton, silently diverging for the rest of the process.

    Only the `argotech.*` entries are touched — never third-party modules. A heavy dependency
    (numpy, joblib, torch) that this import pulls in for the first time has to stay imported: numpy
    in particular cannot be cleanly "unimported" once its C extension has initialised, and removing
    it from `sys.modules` here just so a later import looks fresh makes numpy's own reload guard
    trip for real on the next `import numpy` (`UserWarning: The NumPy module was reloaded`), which
    goes on to break scipy/sklearn's lazy import in an unrelated test. Restoring only the
    `argotech.*` slice avoids that entirely: every module this import pulls in that was already
    loaded stays exactly as it was.
    """
    before = {m for m in sys.modules if m.startswith("argotech")}
    original = {m: sys.modules[m] for m in before}
    try:
        for mod in before:
            del sys.modules[mod]
        importlib.import_module(target)
        return {m for m in sys.modules if m.startswith("argotech")}
    finally:
        for mod in [m for m in sys.modules if m.startswith("argotech")]:
            del sys.modules[mod]
        sys.modules.update(original)


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
