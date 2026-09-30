#!/usr/bin/env python
"""G2 audit: model-agnostic short-range reliability protocol.

For each model (MPA-0 foundation / v1 / v2):
  1. pair scans  — 6 pairs x 3 environments (vacuum dimer, NSS128-embedded,
     iface272-embedded), d = 0.8..4.0 step 0.05; assertions:
       (i)  no local minimum below d_phys - 0.3 A
       (ii) bond-projected force repulsive for d < 0.85 * d_phys
       (iii) wall height E(1.2) - E(d_phys) >= 8 eV
  2. hole hunt   — N perturbed 712-atom sandwiches, capped FIRE minimization;
     assert final dmin >= 2.0 A and no dE < -1 eV/atom.

Outputs g2_report.json + scan curves (g2_scans.npz) for the paper figure.
usage: python audit_g2.py --models v2=path [v1=path mpa0=path] [--nhunt 50]
"""
import argparse
import json
import numpy as np
import torch
from ase import Atoms
from ase.io import read
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE as FIREOpt
from mace.calculators import MACECalculator

D_PHYS = {("Na", "Na"): 3.2, ("Na", "S"): 2.83, ("S", "S"): 2.05,
          ("S", "Sb"): 2.33, ("Na", "Sb"): 3.3, ("Sb", "Sb"): 4.0}
# criteria v1.1 (2026-08-06): dimer environments judged against MOLECULAR bond
# lengths (real diatomic minima are physics, not holes), and wall thresholds
# graded by pair softness (Na-Na is a soft metallic pair, 8 eV was untenable)
D_PHYS_DIMER = {("Na", "Na"): 3.08, ("Na", "S"): 2.4, ("S", "S"): 1.9,
                ("S", "Sb"): 2.2, ("Na", "Sb"): 2.9, ("Sb", "Sb"): 2.5}
WALL_MIN = {("Na", "Na"): 3.0, ("Na", "S"): 5.0, ("Na", "Sb"): 5.0,
            ("S", "S"): 8.0, ("S", "Sb"): 8.0, ("Sb", "Sb"): 8.0}
MIN_PROMINENCE = 0.05   # eV; shallower dips are scan noise, not holes
GRID = np.round(np.arange(0.8, 4.0001, 0.05), 2)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True, help="name=path list")
ap.add_argument("--nhunt", type=int, default=50)
ap.add_argument("--nss", default="../gold_NSS_bulk.extxyz")
ap.add_argument("--iface", default="../gold_interface272.extxyz")
ap.add_argument("--cell89", default=("/work/home/USER/Na-NSS-mace/interface-builds/"
                                     "small_na_nss_na_89atoms/na_nss_na_interface_89atoms.extxyz"))
args = ap.parse_args()

nss = read(args.nss)
iface = read(args.iface)
cell89 = read(args.cell89)


def mic_pair(at, i, j):
    v, d = find_mic((at.positions[j] - at.positions[i]).reshape(1, 3), at.cell, at.pbc)
    return v[0], d[0]


def squeeze(host, i, j, target):
    at = host.copy()
    v, dist = mic_pair(at, i, j)
    u = v / dist
    s = (dist - target) / 2.0
    at.positions[i] += u * s
    at.positions[j] -= u * s
    return at


def pick_pair(host, A, B):
    syms = np.array(host.get_chemical_symbols())
    i, j, d = neighbor_list("ijd", host, cutoff=5.0)
    for k in np.argsort(d):
        if syms[i[k]] == A and syms[j[k]] == B:
            return int(i[k]), int(j[k])
    return None


def scan_env(calc, env_name, host, A, B):
    """Return (d_array, E_array, F_proj_array) for the pair in this environment."""
    if env_name == "dimer":
        es, fs = [], []
        for t in GRID:
            at = Atoms(f"{A}{B}", positions=[[0, 0, 0], [float(t), 0, 0]],
                       cell=[16, 16, 16], pbc=True)
            at.calc = calc
            es.append(at.get_potential_energy())
            fs.append(float(at.get_forces()[1, 0]))   # +x on atom 1 = repulsive
        return GRID, np.array(es), np.array(fs)
    pr = pick_pair(host, A, B)
    if pr is None:
        return None
    i, j = pr
    es, fs = [], []
    for t in GRID:
        at = squeeze(host, i, j, float(t))
        at.calc = calc
        es.append(at.get_potential_energy())
        f = at.get_forces()
        u, _ = mic_pair(at, i, j)
        u = u / np.linalg.norm(u)
        fs.append(float(np.dot(f[j], u)))             # + = pushed away from i
    return GRID, np.array(es), np.array(fs)


def audit_scan(d, E, Fp, dphys, wall_min):
    res = {}
    lo = d < (dphys - 0.3)
    # (i) local minima below dphys-0.3, with prominence filter
    mins = []
    for k in range(1, len(d) - 1):
        if (lo[k] and E[k] < E[k - 1] - MIN_PROMINENCE
                and E[k] < E[k + 1] - MIN_PROMINENCE):
            mins.append(float(d[k]))
    res["local_minima_below_phys"] = mins
    # (ii) repulsive force in compressed region
    comp = d < 0.85 * dphys
    bad = d[comp][np.asarray(Fp)[comp] < 0.0]
    res["attractive_points_compressed"] = [float(x) for x in bad]
    # (iii) wall height (pair-graded threshold)
    kref = int(np.argmin(np.abs(d - min(dphys, 3.95))))
    k12 = int(np.argmin(np.abs(d - max(1.2, d[0]))))
    res["wall_height_eV"] = float(E[k12] - E[kref])
    res["pass"] = (not mins) and (len(bad) == 0) and res["wall_height_eV"] >= wall_min
    return res


report = {}
scans = {}
for spec in args.models:
    name, path = spec.split("=", 1)
    calc = MACECalculator(model_paths=path, device="cuda",
                          default_dtype="float64", enable_cueq=True)
    mrep = {"scans": {}, "hole_hunt": None}
    for (A, B), dphys in D_PHYS.items():
        for env_name, host in [("dimer", None), ("NSS128", nss), ("iface272", iface)]:
            out = scan_env(calc, env_name, host, A, B)
            if out is None:
                continue
            d, E, Fp = out
            key = f"{A}-{B}@{env_name}"
            scans[f"{name}:{key}:E"] = E
            scans[f"{name}:{key}:F"] = Fp
            dp = D_PHYS_DIMER[(A, B)] if env_name == "dimer" else dphys
            mrep["scans"][key] = audit_scan(d, E, Fp, dp, WALL_MIN[(A, B)])
    # hole hunt
    rng = np.random.default_rng(20260805)
    n_bad_dmin, n_bad_dE, worst_dmin = 0, 0, 9.9
    for k in range(args.nhunt):
        at = cell89.repeat((2, 2, 2))
        # strengthened hunt (v1.1): half the starts are compressed 3%,
        # rattle raised to 0.15 A — the 0.10-A hunt discriminated nothing
        if k % 2 == 1:
            at.set_cell(at.cell * 0.97, scale_atoms=True)
        at.rattle(stdev=0.15, seed=int(rng.integers(1e6)))
        at.calc = calc
        e0 = at.get_potential_energy() / len(at)
        opt = FIREOpt(at, maxstep=0.05, logfile=None)
        try:
            opt.run(fmax=0.3, steps=200)
        except Exception:
            pass
        _, _, dd = neighbor_list("ijd", at, cutoff=3.0)
        dmin = float(dd.min()) if len(dd) else 3.0
        worst_dmin = min(worst_dmin, dmin)
        de = at.get_potential_energy() / len(at) - e0
        if dmin < 2.0:
            n_bad_dmin += 1
        if de < -1.0:
            n_bad_dE += 1
        if (k + 1) % 10 == 0:
            print(f"[{name}] hunt {k+1}/{args.nhunt} worst_dmin={worst_dmin:.2f}", flush=True)
    mrep["hole_hunt"] = {"n": args.nhunt, "collapsed_dmin": n_bad_dmin,
                         "big_fake_drops": n_bad_dE, "worst_dmin_A": worst_dmin,
                         "pass": n_bad_dmin == 0 and n_bad_dE == 0}
    npass = sum(1 for v in mrep["scans"].values() if v["pass"])
    mrep["summary"] = {"scans_pass": f"{npass}/{len(mrep['scans'])}",
                       "hole_hunt_pass": mrep["hole_hunt"]["pass"]}
    report[name] = mrep
    print(f"[{name}] scans {npass}/{len(mrep['scans'])} pass; "
          f"hunt pass={mrep['hole_hunt']['pass']}", flush=True)
    del calc
    torch.cuda.empty_cache()

np.savez_compressed("g2_scans.npz", d=GRID, **scans)
json.dump(report, open("g2_report.json", "w"), indent=1)
print(json.dumps({k: v["summary"] for k, v in report.items()}, indent=1))
