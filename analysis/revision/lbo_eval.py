#!/usr/bin/env python
"""Sixth-system accuracy: LPS-LBO model on its 200-frame test set.

Model file is named Li2CO3-LiF_* (copy residue) but log-verified to be the
Li3PS4-Li3B11O18 model (atomic numbers [3,5,8,15,16] = Li,B,O,P,S).
"""
import json
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

MODEL = "/work/home/USER/LPS-LBO/results1/Li2CO3-LiF_stagetwo.model"
TEST = "/work/home/USER/LPS-LBO/soap_test_200.xyz"

calc = MACECalculator(model_paths=MODEL, device="cuda",
                      default_dtype="float64", enable_cueq=True)
de, df = [], []
for at in read(TEST, ":"):
    e0 = float(at.info["dft_energy"])
    f0 = np.asarray(at.arrays["dft_forces"], float)
    a = at.copy()
    a.calc = calc
    de.append((a.get_potential_energy() - e0) / len(a) * 1000)
    df.append(((a.get_forces() - f0) * 1000).ravel())
de = np.array(de)
df = np.concatenate(df)
out = {"model": MODEL, "test": TEST, "n_frames": int(de.size),
       "note": "misnamed file; log-verified Li/B/O/P/S",
       "E_rmse_meV_atom": round(float(np.sqrt((de ** 2).mean())), 3),
       "F_rmse_meV_A": round(float(np.sqrt((df ** 2).mean())), 2)}
json.dump(out, open("lbo_eval.json", "w"), indent=1)
print(out, flush=True)
