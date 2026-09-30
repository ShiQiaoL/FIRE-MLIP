#!/usr/bin/env python
"""Module D: LiCl-GaF3 three-point transferability case (Fig 6b data).

Points: (1) in-distribution 10-frame CP2K test set;
        (2) OOD-SR frame; (3) OOD-LR frame (368 atoms, F_DFT labels).
Evaluates the three LiCl-GaF3 training iterations (results/results1/results2)
plus the CACE-LR baseline already stored in the OOD files (F_CACE vs F_DFT).
"""
import json
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
import torch

R = "/work/home/USER/LiCl-GaF3/remove-errstructure"
MODELS = {
    "resA_ep375": f"{R}/results/LiCl-GaF3_stagetwo.model",
    "resB_ep1500": f"{R}/results1/LiCl-GaF3_stagetwo.model",
    "resC_ep375": f"{R}/results2/LiCl-GaF3_stagetwo.model",
}
TEST = f"{R}/test.xyz"
OOD = {"OOD_SR": "OOD-SR.xyz", "OOD_LR": "OOD-LR.xyz"}


def frms(d):
    return round(float(np.sqrt((np.asarray(d, float) ** 2).mean())) * 1000, 2)


out = {"cace_baseline": {}, "models": {}}

# CACE-LR baseline from stored labels (no GPU)
for tag, path in OOD.items():
    at = read(path)
    fd, fc = at.arrays["F_DFT"], at.arrays["F_CACE"]
    out["cace_baseline"][tag] = {
        "n_atoms": len(at),
        "F_rmse_meV_A": frms(fc - fd),
        "F_maxerr_meV_A": round(float(np.abs(fc - fd).max()) * 1000, 2),
    }
print("[D] CACE baseline:", out["cace_baseline"], flush=True)

test_frames = read(TEST, ":")
for name, mp in MODELS.items():
    entry = {"model": mp}
    try:
        calc = MACECalculator(model_paths=mp, device="cuda",
                              default_dtype="float64", enable_cueq=True)
        de, df = [], []
        for at in test_frames:
            e0 = float(at.info["cp2k_energy"])
            f0 = np.asarray(at.arrays["cp2k_forces"], float)
            a = at.copy()
            a.calc = calc
            de.append((a.get_potential_energy() - e0) / len(a) * 1000)
            df.append((a.get_forces() - f0).ravel())
        entry["indist"] = {
            "n_frames": len(test_frames),
            "E_rmse_meV_atom": round(float(np.sqrt((np.array(de) ** 2).mean())), 3),
            "F_rmse_meV_A": frms(np.concatenate(df)),
        }
        for tag, path in OOD.items():
            at = read(path)
            fd = np.asarray(at.arrays["F_DFT"], float)
            a = at.copy()
            a.calc = calc
            fp = a.get_forces()
            entry[tag] = {
                "F_rmse_meV_A": frms(fp - fd),
                "F_maxerr_meV_A": round(float(np.abs(fp - fd).max()) * 1000, 2),
                "E_pred_eV": round(float(a.get_potential_energy()), 6),
            }
        del calc
        torch.cuda.empty_cache()
        print(f"[D] {name}: {entry['indist']} | SR {entry['OOD_SR']['F_rmse_meV_A']} "
              f"| LR {entry['OOD_LR']['F_rmse_meV_A']}", flush=True)
    except Exception as exc:
        entry["error"] = str(exc)
        print(f"[D] {name} FAILED: {exc}", flush=True)
    out["models"][name] = entry
    json.dump(out, open("pkg_D_results.json", "w"), indent=1)

print("D_DONE", flush=True)
