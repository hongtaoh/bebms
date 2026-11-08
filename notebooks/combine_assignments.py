#!/usr/bin/env python3
"""
Combine the per-participant assignment CSVs written by the ADNI notebooks into
one file for the survival validation.

Run after:
  2025-09-06-adni-bebms.ipynb              -> bebms_assignments.csv
  2025-09-06-adni-sustain-ordering.ipynb   -> sustain_kde_assignments.csv
                                              sustain_gmm_assignments.csv
                                              (run it once per sustainType)

    python combine_assignments.py

Then, from subtrace_iclr_clean:

    python adni_survival_validation.py \
        --results ~/bebms/notebooks/baseline_assignments.csv \
        --out results/adni/survival_baselines
"""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "baseline_assignments.csv")

files = sorted(glob.glob(os.path.join(HERE, "*_assignments.csv")))
files = [f for f in files if os.path.basename(f) != os.path.basename(OUT)]
if not files:
    sys.exit("no *_assignments.csv found -- run the notebook save cells first")

frames, counts = [], {}
for f in files:
    d = pd.read_csv(f)
    missing = {"experiment", "participant", "predicted_subtype",
               "predicted_stage_continuous"} - set(d.columns)
    if missing:
        sys.exit(f"{os.path.basename(f)} is missing columns: {sorted(missing)}")
    frames.append(d)
    for exp, g in d.groupby("experiment"):
        counts[exp] = (len(g), g["predicted_subtype"].nunique())

out = pd.concat(frames, ignore_index=True)

# every method must cover the same participants, or the comparison is not like-for-like
sizes = {e: n for e, (n, _) in counts.items()}
if len(set(sizes.values())) != 1:
    sys.exit(f"methods cover different numbers of participants: {sizes}")

out.to_csv(OUT, index=False)
print(f"wrote {OUT}")
for exp, (n, k) in sorted(counts.items()):
    print(f"  {exp:16} n={n}  K={k}")
