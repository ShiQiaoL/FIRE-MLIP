#!/usr/bin/env python
"""G3 gate: guard-free pilot MD with sentinels only (no repulsive guard).

Protocol: capped FIRE min -> 3 ps @0.5 fs 100 K -> 5 ps @1 fs 300 K ->
100 ps @2 fs NVT 300 K -> 50 ps @2 fs NVE (drift measurement).
Sentinels abort on T>2500 K or dmin<1.35 A. Logs per-species-pair dmin every ps.
Verdict JSON: g3_report.json.

usage: python g3_pilot.py --model <path> [--structure sandwich_prod.extxyz]
"""
import argparse
import json
import time
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE as FIREOpt
from mace.calculators import MACECalculator

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--structure", default="../03_production/sandwich_prod.extxyz")
ap.add_argument("--temp", type=float, default=300.0)
ap.add_argument("--seed", type=int, default=1, help="velocity/thermostat seed; G3 is a statistical gate")
ap.add_argument("--tag", default="g3")
args = ap.parse_args()
seed_rng = np.random.RandomState(args.seed)

atoms = read(args.structure)
atoms.calc = MACECalculator(model_paths=args.model, device="cuda",
                            default_dtype="float64", enable_cueq=True)
n = len(atoms)
syms = np.array(atoms.get_chemical_symbols())
report = {"model": args.model, "natoms": n, "stages": {}, "pair_dmin_min": {}}
pairmin = {}


def pair_dmins():
    i, j, d = neighbor_list("ijd", atoms, cutoff=3.5)
    out = {}
    for a, b, dd in zip(i, j, d):
        key = "-".join(sorted((syms[a], syms[b])))
        if dd < out.get(key, 9.9):
            out[key] = float(dd)
    return out


def sentinel(stage):
    T = atoms.get_kinetic_energy() / n / (1.5 * units.kB)
    pd = pair_dmins()
    for k, v in pd.items():
        if v < pairmin.get(k, 9.9):
            pairmin[k] = v
    dmin = min(pd.values())
    if T > 2500 or dmin < 1.35:
        write(f"{args.tag}_failed_frame.extxyz", atoms)
        report["verdict"] = {"pass": False, "stage": stage, "T": float(T), "dmin": dmin}
        json.dump(report, open(f"{args.tag}_report.json", "w"), indent=1)
        raise SystemExit(f"G3 SENTINEL TRIP: {stage} T={T:.0f} dmin={dmin:.2f}")


t0 = time.time()
e0 = atoms.get_potential_energy()
opt = FIREOpt(atoms, maxstep=0.05, logfile=f"{args.tag}_min.log")
opt.run(fmax=0.15, steps=500)
i, j, d = neighbor_list("ijd", atoms, cutoff=3.0)
report["stages"]["min"] = {"dE_eV": float(atoms.get_potential_energy() - e0),
                           "fmax": float(np.abs(atoms.get_forces()).max()),
                           "dmin": float(d.min()), "converged": bool(opt.converged())}
print("min done:", report["stages"]["min"], flush=True)
# per-pair floor: S-S may legitimately bond (~2.05 A covalent); everything else
# below 1.95 A after minimization indicates collapse
pd0 = pair_dmins()
for k, v in pd0.items():
    floor = 1.80 if k == "S-S" else 1.95
    assert v >= floor, f"minimization collapsed {k} to {v:.2f} A (floor {floor})"

MaxwellBoltzmannDistribution(atoms, temperature_K=100.0, rng=seed_rng)
Stationary(atoms)
for label, dt_fs, T, ps, fric in [("warm100K", 0.5, 100.0, 3, 0.05),
                                  ("warm300K", 1.0, args.temp, 5, 0.05),
                                  ("nvt300K", 2.0, args.temp, 100, 0.02)]:
    steps = int(ps * 1000 / dt_fs)
    dyn = Langevin(atoms, timestep=dt_fs * units.fs, temperature_K=T,
                   friction=fric, rng=seed_rng)
    dyn.attach(lambda label=label: sentinel(label), interval=int(500 / dt_fs))
    Ts = []
    dyn.attach(lambda: Ts.append(atoms.get_kinetic_energy() / n / (1.5 * units.kB)),
               interval=int(1000 / dt_fs))
    dyn.run(steps)
    report["stages"][label] = {"T_mean": float(np.mean(Ts)), "T_std": float(np.std(Ts))}
    print(label, report["stages"][label], flush=True)

# NVE drift
dyn = VelocityVerlet(atoms, timestep=2.0 * units.fs)
E, tps = [], []
dyn.attach(lambda: (E.append(atoms.get_total_energy()),
                    tps.append(dyn.nsteps * 2.0 / 1000.0)), interval=250)
dyn.attach(lambda: sentinel("nve"), interval=250)
dyn.run(25000)   # 50 ps
E, tps = np.array(E), np.array(tps)
slope = float(np.polyfit(tps, E, 1)[0])           # eV per ps
drift = slope / n * 1000 * 1000                   # meV/atom/ns
report["stages"]["nve"] = {"drift_meV_atom_ns": drift,
                           "E_span_meV_atom": float((E.max() - E.min()) / n * 1000)}
report["pair_dmin_min"] = pairmin
pairs_ok = all(v >= (1.75 if k == "S-S" else 1.90) for k, v in pairmin.items())
report["verdict"] = {"pass": abs(drift) <= 5.0 and pairs_ok,
                     "drift_meV_atom_ns": drift, "pair_dmin": pairmin,
                     "wall_h": (time.time() - t0) / 3600}
json.dump(report, open(f"{args.tag}_report.json", "w"), indent=1)
write(f"{args.tag}_final_frame.extxyz", atoms)
print("G3 VERDICT:", json.dumps(report["verdict"], indent=1), flush=True)
