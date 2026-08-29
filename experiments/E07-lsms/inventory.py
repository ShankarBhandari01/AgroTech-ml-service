"""E07 step 1 — what is actually in the GHS-Panel Wave 5 release, and at what grain.

The CSV export carries no variable labels (the .dta export does), so column names are bare question
codes: `ag1`, `sa3iq3`. Nothing here guesses what a code means. This step establishes only what can
be established from structure alone, which is what determines whether E07 is feasible at all:

  * the grain of every file -- which key combination is unique, so joins cannot silently fan out
  * how many households and plots each module reaches
  * which files share a key with which, i.e. the join skeleton
  * where the sample sizes actually land, since E07's power is bounded by the smallest module

Run: venv/bin/python3 experiments/E07-lsms/inventory.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path("data/lsms/NGA_2023_GHSP-W5_v01_M_CSV")
OUT = Path("experiments/E07-lsms/inventory.json")

# Candidate keys, most specific first. LSMS files are keyed by some prefix of these.
KEYS = ["hhid", "plotid", "cropcode", "indiv", "ea", "lga", "state", "zone"]


def grain(df: pd.DataFrame) -> tuple[list[str], bool]:
    """The shortest key combination that is unique, and whether one was found."""
    present = [k for k in KEYS if k in df.columns]
    for n in range(1, len(present) + 1):
        for i in range(len(present) - n + 1):
            combo = present[i:i + n]
            if not df.duplicated(combo).any():
                return combo, True
    return present, False


def main() -> None:
    files = sorted(p for p in ROOT.rglob("*.csv") if ".ipynb_checkpoints" not in str(p))
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:                      # noqa: BLE001
            rows.append({"file": str(f.relative_to(ROOT)), "error": str(e)[:120]})
            continue
        g, unique = grain(df)
        rows.append({
            "file": str(f.relative_to(ROOT)),
            "visit": "post_planting" if "planting" in f.name else "post_harvest",
            "module": f.parent.name,
            "rows": len(df),
            "cols": len(df.columns),
            "grain": g,
            "grain_is_unique": unique,
            "n_hh": int(df["hhid"].nunique()) if "hhid" in df else None,
            "n_ea": int(df["ea"].nunique()) if "ea" in df else None,
            "n_plot": int(df.groupby(["hhid", "plotid"]).ngroups) if {"hhid", "plotid"} <= set(df.columns) else None,
            "columns": list(df.columns),
        })
        print(f"  {f.relative_to(ROOT)}: {len(df):>7} rows x {len(df.columns):>3} cols  "
              f"grain={'+'.join(g) if g else '-'}{'' if unique else ' (NOT unique)'}", flush=True)

    inv = pd.DataFrame([r for r in rows if "error" not in r])
    print("\n=== reach, by module ===")
    print(inv.groupby(["visit", "module"]).agg(files=("file", "count"), max_hh=("n_hh", "max"),
                                               max_ea=("n_ea", "max"), max_plot=("n_plot", "max")).to_string())
    print("\n=== the widest household and plot rosters (where identity lives) ===")
    for col, label in [("n_hh", "households"), ("n_plot", "plots")]:
        top = inv.dropna(subset=[col]).nlargest(4, col)[["file", "rows", col]]
        print(f"\n most {label}:\n{top.to_string(index=False)}")
    print("\n=== files with NO usable unique grain (joins here can fan out) ===")
    bad = inv[~inv.grain_is_unique]
    print(f"  {len(bad)} of {len(inv)}")
    for r in bad.head(8).itertuples():
        print(f"    {r.file}  (grain tried: {'+'.join(r.grain) if r.grain else '-'})")

    OUT.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
