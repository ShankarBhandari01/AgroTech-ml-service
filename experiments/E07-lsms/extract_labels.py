"""E07 — pull variable and value labels out of the Stata export, and find the coping factors.

Why this exists: the CSV export drops all metadata, so columns arrive as bare question codes
(`ag1`, `sa3iq3`, `s11b1q43`). Six of `domain.risk.COPING_FACTORS` cannot be identified from codes
without guessing, and a wrong guess would silently produce a wrong vulnerability term. The Stata
(.dta) export carries `variable_labels()` (the question text) and `value_labels()` (the code->answer
mapping), which resolves every one of them.

Point it at the STATA download:
    venv/bin/python3 experiments/E07-lsms/extract_labels.py data/lsms/NGA_2023_GHSP-W5_v01_M_STATA

Writes `labels.json` (every variable, every file) and prints ranked candidates per coping factor.
Nothing here decides a mapping -- it ranks candidates by how well the question text matches, and a
human confirms. That is deliberate: the whole reason for this step is to stop guessing.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path("experiments/E07-lsms")

# The seven signals `risk.assess_vulnerability` consumes, and the words a questionnaire would use.
# `crop_diversity` is already solved structurally (count of `cropcode`), so it is not searched here.
COPING_PATTERNS = {
    "has_irrigation":       r"\birrigat",
    "has_extension_access": r"extension|advisory|agricultur\w+ (advice|information)",
    "received_credit":      r"\bcredit\b|\bloan\b|borrow",
    "used_fertilizer":      r"fertili[sz]er|\burea\b|\bnpk\b",
    "asset_score":          r"\bassets?\b|durable|own.*(radio|tv|phone|bicycle|motorcycle)",
    "market_access_score":  r"market|distance to.*(market|road)|sell",
}
# Exposure inputs already CONFIRMED from the CSVs; searched only to cross-check the labels agree.
CONFIRM_PATTERNS = {
    "area_ha (SR_hect)":  r"area|size of (the )?plot|hectare",
    "harvest quantity":   r"quantity harvest|how much.*harvest",
    "harvest value":      r"value.*(harvest|produce)|worth",
}


def read_labels(path: Path) -> tuple[dict, dict]:
    """(variable_labels, value_labels) for one .dta, or ({}, {}) if unreadable."""
    try:
        with pd.io.stata.StataReader(str(path)) as r:
            return r.variable_labels(), r.value_labels()
    except Exception as e:                                   # noqa: BLE001
        print(f"  ! {path.name}: {e}", file=sys.stderr)
        return {}, {}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1])
    if not root.exists():
        raise SystemExit(f"not found: {root}\nDownload the _STATA build, not _CSV.")

    files = sorted(root.rglob("*.dta"))
    print(f"{len(files)} .dta files under {root}\n")
    if not files:
        raise SystemExit("No .dta files. This looks like the CSV export -- get the _STATA one.")

    catalogue: dict[str, dict] = {}
    for f in files:
        var, val = read_labels(f)
        if not var:
            continue
        # Stata value-label keys arrive as numpy int32, which json cannot serialise. Coerce to
        # plain ints rather than str: the codes are numeric and comparing "1" to 1 downstream is a
        # silent mismatch waiting to happen.
        val = {name: {int(k): str(v) for k, v in mapping.items()}
               for name, mapping in val.items()}
        catalogue[str(f.relative_to(root))] = {"variables": var, "value_labels": val}
    (OUT/"labels.json").write_text(json.dumps(catalogue, indent=1))
    n_vars = sum(len(v["variables"]) for v in catalogue.values())
    print(f"catalogued {n_vars} labelled variables across {len(catalogue)} files -> {OUT}/labels.json\n")

    def search(patterns: dict[str, str], header: str) -> None:
        print(f"=== {header} ===")
        for target, pat in patterns.items():
            rx = re.compile(pat, re.I)
            hits = [(f, code, text) for f, blob in catalogue.items()
                    for code, text in blob["variables"].items() if text and rx.search(text)]
            print(f"\n{target}  ({len(hits)} candidates)")
            for f, code, text in hits[:6]:
                print(f"   {code:16s} {f.split('/')[-1]:34s} {text[:78]}")
            if not hits:
                print("   none -- widen the pattern, or the module may be absent from this wave")

    search(COPING_PATTERNS, "COPING FACTORS (unidentified; confirm before use)")
    search(CONFIRM_PATTERNS, "CROSS-CHECK of mappings already confirmed from the CSVs")

    print("\nNext: confirm each code against its question text, then add it to a mapping module.")
    print("Do NOT wire a candidate in on pattern match alone -- that is the guess this step exists")
    print("to avoid. Check the value labels too: a 1/2 Yes/No coding read as 1/0 inverts the signal.")


if __name__ == "__main__":
    main()
