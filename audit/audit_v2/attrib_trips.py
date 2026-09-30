#!/usr/bin/env python
"""Null-model test data for the 0/30 deployment post-mortem.

For every R1 trip frame (and the three 08-01 evidence frames): histogram of
element pairs in sub-physical contact (d < 1.35 A, the sentinel criterion),
plus the global-minimum pair. Null model reference: random-pair probabilities
proportional to composition products n_A*n_B.

CPU-only, login node. Output: trips_attrib.json.
"""
import glob
import json
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

SENT_D = 1.35
FILES = sorted(glob.glob("r1s1_trip_*.extxyz")) + sorted(
    glob.glob("../03_production/evidence/*_20260801.extxyz"))

rows = []
comp_done = None
for f in FILES:
    try:
        at = read(f, -1)
    except Exception as exc:
        rows.append({"file": f, "error": str(exc)})
        continue
    syms = np.array(at.get_chemical_symbols())
    if comp_done is None:
        u, c = np.unique(syms, return_counts=True)
        comp = dict(zip(u.tolist(), c.tolist()))
        # unordered-pair null weights ~ nA*nB (A!=B) or nA*(nA-1)/2
        w = {}
        keys = sorted(comp)
        for a in range(len(keys)):
            for b in range(a, len(keys)):
                A, Bk = keys[a], keys[b]
                w["-".join((A, Bk))] = (comp[A] * (comp[A] - 1) / 2
                                       if A == Bk else comp[A] * comp[Bk])
        tot = sum(w.values())
        null = {k: round(v / tot, 4) for k, v in w.items()}
        comp_done = {"composition": comp, "null_pair_prob": null}
    i, j, d = neighbor_list("ijd", at, cutoff=SENT_D)
    hist = {}
    seen = set()
    for k in range(len(d)):
        key = tuple(sorted((int(i[k]), int(j[k]))))
        if key in seen:
            continue
        seen.add(key)
        pr = "-".join(sorted((syms[i[k]], syms[j[k]])))
        hist[pr] = hist.get(pr, 0) + 1
    gmin_pair = None
    if len(d):
        k = int(np.argmin(d))
        gmin_pair = {"pair": "-".join(sorted((syms[i[k]], syms[j[k]]))),
                     "d": round(float(d[k]), 3)}
    rows.append({"file": f.split("/")[-1], "n_atoms": len(at),
                 "n_subphys_contacts": len(seen),
                 "contact_pairs": hist, "global_min": gmin_pair})
    print(f"{f.split('/')[-1]}: {len(seen)} contacts {hist} "
          f"gmin={gmin_pair}", flush=True)

json.dump({"sentinel_d": SENT_D, **(comp_done or {}), "frames": rows},
          open("trips_attrib.json", "w"), indent=1)
print("TRIPS_ATTRIB_DONE", flush=True)
