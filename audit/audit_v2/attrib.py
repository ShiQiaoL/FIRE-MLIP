#!/usr/bin/env python
"""Coin 3: element-pair attribution of the robustness-grid warning events.

For every saved trajectory frame (72 runs x ~50 frames): global min distance,
which element pair produced it, and per-pair minimum distances. Separates
covalent-bond thermal vibration (C-O, B-O, P-O, S-P...) from genuine
non-bonded grazing (Li-Li, Na-Na, Li-O close approach...).

CPU-only. Output: attrib_results.json (one row per run).
"""
import glob
import json
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

WARN_D = 1.2
rows = []
for f in sorted(glob.glob("md_*.extxyz")):
    tag = f[3:-7]
    frames = read(f, ":")
    pair_min = {}
    dmin_series = []
    warn_pairs = {}
    for at in frames:
        syms = np.array(at.get_chemical_symbols())
        i, j, d = neighbor_list("ijd", at, cutoff=3.0)
        if not len(d):
            continue
        k = int(np.argmin(d))
        gmin = float(d[k])
        gpair = "-".join(sorted((syms[i[k]], syms[j[k]])))
        dmin_series.append(gmin)
        if gmin < WARN_D:
            warn_pairs[gpair] = warn_pairs.get(gpair, 0) + 1
        for a in range(len(d)):
            key = "-".join(sorted((syms[i[a]], syms[j[a]])))
            if key not in pair_min or d[a] < pair_min[key]:
                pair_min[key] = float(d[a])
    rows.append({
        "run": tag, "n_frames": len(frames),
        "global_dmin": round(min(dmin_series), 3) if dmin_series else None,
        "n_frames_below_1p2": sum(1 for x in dmin_series if x < WARN_D),
        "warn_pair_counts": warn_pairs,
        "per_pair_min": {k: round(v, 3) for k, v in sorted(pair_min.items())},
    })
    print(f"{tag}: gmin={rows[-1]['global_dmin']} "
          f"warn_frames={rows[-1]['n_frames_below_1p2']} "
          f"pairs={warn_pairs}", flush=True)

json.dump(rows, open("attrib_results.json", "w"), indent=1)
print("ATTRIB_DONE", flush=True)
