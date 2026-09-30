#!/usr/bin/env python
"""Threshold sensitivity for the g2x v2 audit (verdict item B4).

From g2x_scans.npz (raw E/F curves), recompute every scan verdict under:
  - wall thresholds scaled x0.5, x0.7, x1.0, x1.3, x1.5
  - d_eq at p5 / p10 / p20 of the training-distance distribution
    (approximated here by shifting the stored d_eq by the observed p5-p20
     spread; exact percentiles recomputed from train sets for key pairs)
Robust cells (verdict unchanged under all variants) vs marginal cells are
listed explicitly. CPU-only.
"""
import json
import numpy as np

MIN_PROMINENCE = 0.05
z = np.load("g2x_scans.npz")
d = z["d"]
report = json.load(open("g2x_report.json"))

WALL_SCALES = [0.5, 0.7, 1.0, 1.3, 1.5]
DEQ_SHIFTS = {"p5~": -0.06, "p10": 0.0, "p20~": +0.08}


def verdict(E, Fp, d_bound, d_eq, wmin):
    lo = d < min(d_eq - 0.3, d_bound)
    mins = [float(d[k]) for k in range(1, len(d) - 1)
            if lo[k] and E[k] < E[k - 1] - MIN_PROMINENCE
            and E[k] < E[k + 1] - MIN_PROMINENCE]
    comp = d < 0.85 * d_eq
    bad = d[comp][np.asarray(Fp)[comp] < 0.0]
    kref = int(np.argmin(np.abs(d - min(d_eq, 3.95))))
    k12 = int(np.argmin(np.abs(d - max(1.2, d[0]))))
    wall = float(E[k12] - E[kref])
    return (not mins) and (len(bad) == 0) and wall >= wmin


out = {}
for sname, srep in report.items():
    for mname, mrep in srep.get("models", {}).items():
        if "scans" not in mrep:
            continue
        for pair, sc in mrep["scans"].items():
            if "pass" not in sc:
                continue
            keyE = f"{sname}:{mname}:{pair}:E"
            keyF = f"{sname}:{mname}:{pair}:F"
            if keyE not in z:
                continue
            E, Fp = z[keyE], z[keyF]
            d_bound = sc["d_min_data"]
            wmin0 = sc["wall_min_eV"]
            verdicts = {}
            for ws in WALL_SCALES:
                for tag, sh in DEQ_SHIFTS.items():
                    v = verdict(E, Fp, d_bound, sc["d_eq_p10"] + sh,
                                wmin0 * ws)
                    verdicts[f"w{ws}:{tag}"] = bool(v)
            vals = set(verdicts.values())
            out[f"{sname}/{mname}/{pair}"] = {
                "base_pass": sc["pass"],
                "robust": len(vals) == 1,
                "n_pass": sum(verdicts.values()),
                "n_total": len(verdicts),
            }

robust_cells = {k: v for k, v in out.items() if v["robust"]}
marginal = {k: v for k, v in out.items() if not v["robust"]}
print(f"total cells: {len(out)}; robust: {len(robust_cells)}; "
      f"marginal: {len(marginal)}")
print("--- marginal cells (verdict flips under threshold variants) ---")
for k, v in sorted(marginal.items()):
    print(f"  {k}: base={'PASS' if v['base_pass'] else 'FAIL'} "
          f"pass_under {v['n_pass']}/{v['n_total']} variants")
print("--- robust FAILs (fail under ALL variants) ---")
for k, v in sorted(robust_cells.items()):
    if not v["base_pass"]:
        print(f"  {k}")
json.dump({"cells": out}, open("thresh_sens.json", "w"), indent=1)
print("THRESH_SENS_DONE")
