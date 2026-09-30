#!/usr/bin/env python
"""G3 diagnostic rerun: catch the seed event of the NVT-stage explosion.

Differences vs g3_pilot:
  - rolling snapshot buffer (every 25 steps = 0.05 ps, keep last 60 = 3 ps)
  - warning events: first time any species pair dips below its warning
    distance, save the frame (throttled to >=0.2 ps apart, max 40) and keep
    running -- these are the labelable pre-collapse configurations
  - catastrophic sentinel (T>2500 K or dmin<1.30) dumps the whole buffer

usage: python g3_diag.py --model <path>
"""
import argparse
import collections
import json
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE as FIREOpt
from mace.calculators import MACECalculator

WARN_D = {"Na-Na": 2.45, "Na-S": 2.05, "Na-Sb": 2.55, "S-S": 1.88,
          "S-Sb": 2.05, "Sb-Sb": 2.65}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--structure", default="../03_production/sandwich_prod.extxyz")
args = ap.parse_args()

atoms = read(args.structure)
atoms.calc = MACECalculator(model_paths=args.model, device="cuda",
                            default_dtype="float64", enable_cueq=True)
n = len(atoms)
syms = np.array(atoms.get_chemical_symbols())
buffer = collections.deque(maxlen=60)
warnings_log = []
last_warn_step = -1000
warn_count = 0
events = json.load(open("g3_diag_events.json")) if False else []


def pair_key(a, b):
    return "-".join(sorted((syms[a], syms[b])))


def snapshot(step, tag):
    fr = atoms.copy()
    fr.info["md_step"] = int(step)
    fr.info["tag"] = tag
    return fr


def check(dyn, stage):
    global last_warn_step, warn_count
    step = dyn.nsteps
    buffer.append(snapshot(step, f"{stage}_buf"))
    T = atoms.get_kinetic_energy() / n / (1.5 * units.kB)
    i, j, d = neighbor_list("ijd", atoms, cutoff=2.8)
    dmin_global = float(d.min()) if len(d) else 2.8
    # warning events
    if warn_count < 40 and step - last_warn_step >= 100:
        hits = []
        for a, b, dd in zip(i, j, d):
            k = pair_key(a, b)
            if dd < WARN_D.get(k, 0.0):
                hits.append((k, int(a), int(b), float(dd)))
        if hits:
            hits.sort(key=lambda x: x[3])
            fr = snapshot(step, "warning")
            fr.info["warn_pairs"] = str(hits[:5])
            write(f"g3diag_warn_{warn_count:02d}.extxyz", fr)
            events.append({"step": int(step), "stage": stage, "T": float(T),
                           "pairs": hits[:5]})
            warn_count += 1
            last_warn_step = step
            print(f"WARN#{warn_count} {stage} step={step} T={T:.0f} {hits[:3]}",
                  flush=True)
    # catastrophic
    if T > 2500 or dmin_global < 1.30:
        for kk, fr in enumerate(buffer):
            write(f"g3diag_buffer_{kk:02d}.extxyz", fr)
        write("g3diag_trip.extxyz", atoms)
        json.dump(events, open("g3_diag_events.json", "w"), indent=1)
        raise SystemExit(f"TRIP {stage} step={step} T={T:.0f} dmin={dmin_global:.2f}")


opt = FIREOpt(atoms, maxstep=0.05, logfile="g3diag_min.log")
opt.run(fmax=0.15, steps=500)
MaxwellBoltzmannDistribution(atoms, temperature_K=100.0)
Stationary(atoms)
for label, dt_fs, T, ps, fric in [("warm100K", 0.5, 100.0, 3, 0.05),
                                  ("warm300K", 1.0, 300.0, 5, 0.05),
                                  ("nvt300K", 2.0, 300.0, 100, 0.02)]:
    dyn = Langevin(atoms, timestep=dt_fs * units.fs, temperature_K=T, friction=fric)
    dyn.attach(lambda dyn=dyn, label=label: check(dyn, label), interval=25)
    print(f"stage {label} ...", flush=True)
    dyn.run(int(ps * 1000 / dt_fs))

json.dump(events, open("g3_diag_events.json", "w"), indent=1)
print(f"COMPLETED WITHOUT TRIP; warnings={warn_count}")
