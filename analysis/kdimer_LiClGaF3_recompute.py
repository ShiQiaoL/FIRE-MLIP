"""Recompute the LiCl/GaF3 K-module dimer curves (Ga-Ga, F-Li) with the paper's
Fig. 2 model (iter-two) instead of the 82-frame results1 model used in the revision audit.
Also re-evaluates the foundation model as a sanity check against the archived kdimer_model.json."""
import json, sys, os
import numpy as np
from ase import Atoms
from mace.calculators import MACECalculator

A = "ARCHIVE/dzeshell"
K = "ARCHIVE/mace-prep-20260801/08_kmod"
OUT = "ARCHIVE/kmod-LiClGaF3-itertwo-20260921"
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "itertwo": A + "/LiCl-GaF3/remove-errstructure/iter-two/results/LiCl-GaF3_stagetwo.model",
    "results1": A + "/LiCl-GaF3/remove-errstructure/results1/LiCl-GaF3_stagetwo.model",
    "mpa0": A + "/multi-head-train/mace-mpa-0-medium.model",
}
PAIRS = {"GaGa": ("Ga", "Ga"), "FLi": ("F", "Li")}
DISTS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 1.9, 2.2, 2.5, 2.8, 3.2, 3.6]

dft = json.load(open(K + "/kmod_curves.json"))
old = json.load(open(K + "/kdimer_model.json"))

calcs = {k: MACECalculator(model_paths=v, device="cpu", default_dtype="float64") for k, v in MODELS.items()}
res = {}
for tag, (a, b) in PAIRS.items():
    row = {"d": DISTS}
    for mk, c in calcs.items():
        E = []
        for t in DISTS:
            at = Atoms(f"{a}{b}", positions=[[0, 0, 0], [t, 0, 0]], cell=[16, 16, 16], pbc=True)
            at.calc = c
            E.append(float(at.get_potential_energy()))
        row["E_" + mk] = E
    res[tag] = row

def wall(d, E, at=0.9):
    E = np.array(E); i = d.index(at)
    return float(E[i] - E.min())

report = {}
for tag in PAIRS:
    d = res[tag]["d"]
    dd = [p["d"] for p in dft[tag] if p["converged"]]
    Ed = [p["E0_eV"] for p in dft[tag] if p["converged"]]
    # DFT wall at 0.9 relative to the DFT minimum over the converged grid
    wd = float(np.array(Ed)[dd.index(0.9)] - np.array(Ed).min())
    r = {"DFT_wall_0.9": wd}
    for mk in MODELS:
        w = wall(d, res[tag]["E_" + mk]); r[mk + "_wall_0.9"] = w; r[mk + "_pct_of_DFT"] = 100 * w / wd
    # consistency with the archived audit values
    r["archived_ft(results1)_wall_0.9"] = wall(old[tag]["d"], old[tag]["E_ft"])
    r["archived_mpa0_wall_0.9"] = wall(old[tag]["d"], old[tag]["E_mpa0"])
    report[tag] = r

json.dump({"curves": res, "report": report, "models": MODELS}, open(OUT + "/kdimer_LiClGaF3_itertwo.json", "w"), indent=1)
for tag, r in report.items():
    print(f"== {tag}  DFT wall@0.9 = {r['DFT_wall_0.9']:.2f} eV")
    for mk in MODELS:
        print(f"   {mk:9s} wall = {r[mk+'_wall_0.9']:7.2f} eV  ({r[mk+'_pct_of_DFT']:5.1f}% of DFT)")
    print(f"   archived: results1 {r['archived_ft(results1)_wall_0.9']:.2f}, mpa0 {r['archived_mpa0_wall_0.9']:.2f}")
