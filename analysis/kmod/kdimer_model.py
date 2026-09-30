#!/usr/bin/env python
"""Model-side vacuum dimer curves on the module-K grid (CPU, login node).

Each audited pair is evaluated with its system's fine-tuned model AND the
foundation model, on the same 16 A box / same distances as the DFT singles.
Output: kdimer_model.json.
"""
import json
import numpy as np
from ase import Atoms
from mace.calculators import MACECalculator

B = "/work/home/USER/Na-NSS-mace/new-mace-Na-NSS-20250605/multi-head-train"
MPA0 = f"{B}/2000-fra-2-stress-add-T-NSS/mace-mpa-0-medium.model"
PAIRS = {
    "CO": ("C", "O", "/work/home/USER/Li2CO3-LiF/results1/Li2CO3-LiF_stagetwo.model"),
    "BO": ("B", "O", "/work/home/USER/LPS-LBO/results1/Li2CO3-LiF_stagetwo.model"),
    "GaGa": ("Ga", "Ga", "/work/home/USER/LiCl-GaF3/remove-errstructure/results1/LiCl-GaF3_stagetwo.model"),
    "LiLi": ("Li", "Li", "/work/home/USER/Li-LPSCl/train-numpt-15000/results/Li-LPSCl_stagetwo.model"),
    "SS": ("S", "S", "/work/home/USER/Li-LPSCl/train-numpt-15000/results/Li-LPSCl_stagetwo.model"),
    "FLi": ("F", "Li", "/work/home/USER/LiCl-GaF3/remove-errstructure/results1/LiCl-GaF3_stagetwo.model"),
    "LiO": ("Li", "O", "/work/home/USER/LLZO-SUN/multi-head-train/2000-fra/results/LLZO-SUN_stagetwo.model"),
}
DISTS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 1.9, 2.2, 2.5, 2.8, 3.2, 3.6]

calcs = {}


def calc_of(path):
    if path not in calcs:
        calcs[path] = MACECalculator(model_paths=path, device="cpu",
                                     default_dtype="float64")
    return calcs[path]


out = {}
for tag, (A, Bb, ftpath) in PAIRS.items():
    row = {"elements": [A, Bb], "ft_model": ftpath, "d": DISTS,
           "E_ft": [], "E_mpa0": []}
    for mkey, mpath in [("E_ft", ftpath), ("E_mpa0", MPA0)]:
        c = calc_of(mpath)
        for t in DISTS:
            at = Atoms(f"{A}{Bb}", positions=[[0, 0, 0], [t, 0, 0]],
                       cell=[16, 16, 16], pbc=True)
            at.calc = c
            row[mkey].append(round(float(at.get_potential_energy()), 6))
    out[tag] = row
    print(f"{tag} done", flush=True)

json.dump(out, open("kdimer_model.json", "w"), indent=1)
print("KDIMER_MODEL_DONE", flush=True)
