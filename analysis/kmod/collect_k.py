#!/usr/bin/env python3
"""Harvest module-K dimer DFT singles -> kmod_curves.json.

Reads the sigma->0 energy (E0) from each OSZICAR final line; flags
non-converged runs (NELM exhausted or missing output).
"""
import glob
import json
import os
import re

rows = {}
for d in sorted(glob.glob("k_*/")):
    tag = d.rstrip("/")
    m = re.match(r"k_([A-Za-z]+)_d(\d+)p(\d+)", tag)
    pair, di, df = m.group(1), m.group(2), m.group(3)
    dist = float(f"{di}.{df}")
    entry = {"d": dist}
    osz = os.path.join(d, "OSZICAR")
    ok = False
    if os.path.exists(osz):
        lines = open(osz).read().strip().splitlines()
        for ln in reversed(lines):
            mm = re.search(r"E0= ([-.\dE+]+)", ln)
            if mm:
                entry["E0_eV"] = float(mm.group(1))
                ok = True
                break
        # convergence: last electronic step count < NELM(200)
        nel = [int(x.split()[1]) for x in lines
               if re.match(r"^(DAV|RMM):", x.strip())]
        entry["n_scf_last"] = nel[-1] if nel else None
        entry["converged"] = bool(ok and nel and nel[-1] < 200)
    if not ok:
        entry["error"] = "no E0"
    rows.setdefault(pair, []).append(entry)

for p in rows:
    rows[p].sort(key=lambda r: r["d"])
json.dump(rows, open("kmod_curves.json", "w"), indent=1)
bad = [(p, r["d"]) for p in rows for r in rows[p]
       if not r.get("converged")]
print(f"pairs={list(rows)} total={sum(len(v) for v in rows.values())} "
      f"unconverged={bad}")
