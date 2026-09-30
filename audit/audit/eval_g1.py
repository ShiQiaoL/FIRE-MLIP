#!/usr/bin/env python
"""G1 gate: v1 vs v2 grouped accuracy.

Datasets: original valid.extxyz (300 frames, grouped by natoms; 272 = interface
subset) and repulsion_valid.xyz (grouped by config_type).
Criteria (vs v1 baseline 0.47 / 19.4): original-valid E RMSE <= 0.6 meV/atom,
F RMSE <= 24 meV/A, interface-272 F RMSE <= 28; repulsion groups E <= 5, F <= 100.
"""
import json
import numpy as np
from collections import defaultdict
from ase.io import read
from mace.calculators import MACECalculator

B = "/work/home/USER/Na-NSS-mace/new-mace-Na-NSS-20250605/multi-head-train"
MODELS = {"v1": f"{B}/2000-fra-2-stress-add-T-NSS/results/Na-NSS-2000_stagetwo.model",
          "v2": f"{B}/v2-SR-20260805/results/Na-NSS-v2-SR_stagetwo.model",
          "v21": f"{B}/v21-SR-20260806/results/Na-NSS-v21-SR_stagetwo.model"}
SETS = {"orig_valid": "/work/home/USER/Na-NSS-mace/mace-prep-20260801/02_subset_eval/valid.extxyz",
        "repulsion21_valid": f"{B}/v21-SR-20260806/repulsion21_valid.xyz"}


def ref_of(at):
    r = at.calc.results if at.calc else {}
    e = r.get("free_energy", r.get("energy", at.info.get("free_energy", at.info.get("energy"))))
    f = r.get("forces", at.arrays.get("forces"))
    return float(e), np.asarray(f, float)


def group_of(setname, at):
    if setname == "orig_valid":
        return "interface_272" if len(at) == 272 else f"bulk_{len(at)}"
    return at.info.get("config_type", "unknown")


report = {}
for mname, mpath in MODELS.items():
    calc = MACECalculator(model_paths=mpath, device="cuda",
                          default_dtype="float64", enable_cueq=True)
    for sname, spath in SETS.items():
        groups = defaultdict(lambda: {"de": [], "df": []})
        for at in read(spath, ":"):
            e0, f0 = ref_of(at)
            n = len(at)
            at.calc = calc
            de = (at.get_potential_energy() - e0) / n * 1000
            df = (at.get_forces() - f0) * 1000
            g = groups[group_of(sname, at)]
            g["de"].append(de)
            g["df"].append(df.ravel())
        for gname, g in groups.items():
            de = np.array(g["de"])
            df = np.concatenate(g["df"])
            report[f"{mname}:{sname}:{gname}"] = {
                "n": int(de.size),
                "E_rmse": round(float(np.sqrt((de ** 2).mean())), 3),
                "F_rmse": round(float(np.sqrt((df ** 2).mean())), 2)}
    del calc
    import torch
    torch.cuda.empty_cache()

json.dump(report, open("g1_report.json", "w"), indent=1)
print(f"{'key':48s} {'n':>4s} {'E_RMSE':>8s} {'F_RMSE':>8s}")
for k in sorted(report):
    v = report[k]
    print(f"{k:48s} {v['n']:4d} {v['E_rmse']:8.3f} {v['F_rmse']:8.2f}")
