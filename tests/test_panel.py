"""The manifest is what makes a reported number traceable to the data that produced it.

docs/RESEARCH_SUMMARY.md records a metrics table cited by the README and two source files that
exists in no committed file. A content hash beside every panel is how that stops happening.
"""

from __future__ import annotations

import json

import pandas as pd

from argotech.lab.panel import manifest, write_panel


def _df() -> pd.DataFrame:
    return pd.DataFrame({"site_id": ["A", "A", "B"], "cluster": ["C0", "C0", "C1"],
                         "obs_date": ["2025-01-01", "2025-02-01", "2025-01-01"],
                         "forward_z": [0.1, -0.2, 0.3]})


def test_manifest_describes_the_panel():
    m = manifest(_df())
    assert m["rows"] == 3 and m["sites"] == 2 and m["clusters"] == ["C0", "C1"]
    assert m["date_min"] == "2025-01-01" and m["date_max"] == "2025-02-01"
    assert len(m["content_hash"]) == 64
    assert m["code_version"], "the code version that built the panel must be recorded"


def test_the_hash_is_stable_across_identical_frames():
    assert manifest(_df())["content_hash"] == manifest(_df())["content_hash"]


def test_the_hash_is_stable_across_row_order():
    shuffled = _df().iloc[::-1].reset_index(drop=True)
    assert manifest(_df())["content_hash"] == manifest(shuffled)["content_hash"], \
        "row order is not data; a re-sorted panel is the same panel"


def test_the_hash_changes_when_a_value_changes():
    changed = _df()
    changed.loc[0, "forward_z"] = 0.10001
    assert manifest(_df())["content_hash"] != manifest(changed)["content_hash"]


def test_write_panel_emits_a_sidecar(tmp_path):
    p = write_panel(_df(), tmp_path / "panel.parquet")
    side = json.loads(p.with_suffix(".parquet.manifest.json").read_text())
    assert side["content_hash"] == manifest(_df())["content_hash"]
    assert pd.read_parquet(p).shape == (3, 4)
