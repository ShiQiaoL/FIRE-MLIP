#!/usr/bin/env python
"""Coin 2: hole hunt at n=100 starts (power upgrade from n=20).

Same protocol as audit_g2x v2 hunts: >=300-atom supercell of the system's
test frame, half the starts compressed 3%, rattle 0.15, capped FIRE
(maxstep 0.05, fmax 0.3, 200 steps); collapse = dmin < floor; fake drop =
dE < -1 eV/atom. Floors reproduced verbatim from g2x_report.json (v2).

Per-model checkpoint: hunt100_{sys}_{model}.json (skip if exists).
Run order puts the decisive models first (LBO, then the fake-drop systems).
"""
import json
import os
import numpy as np
import torch
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE as FIREOpt
from mace.calculators import MACECalculator

B = "/work/home/USER/Na-NSS-mace/new-mace-Na-NSS-20250605/multi-head-train"
MPA0 = f"{B}/2000-fra-2-stress-add-T-NSS/mace-mpa-0-medium.model"
NHUNT = 100

SYSTEMS = {
    "LPSLBO": {"host": "/work/home/USER/LPS-LBO/soap_test_200.xyz",
               "floor": 0.861,
               "models": {"ft": "/work/home/USER/LPS-LBO/results1/Li2CO3-LiF_stagetwo.model"}},
    "LiClGaF3": {"host": "/work/home/USER/LiCl-GaF3/remove-errstructure/test.xyz",
                 "floor": 0.272,
                 "models": {"ft": "/work/home/USER/LiCl-GaF3/remove-errstructure/results1/LiCl-GaF3_stagetwo.model"}},
    "Li2CO3LiF": {"host": "/work/home/USER/Li2CO3-LiF/mace_test.xyz",
                  "floor": 0.825,
                  "models": {"ft": "/work/home/USER/Li2CO3-LiF/results1/Li2CO3-LiF_stagetwo.model"}},
    "NaNSS": {"host": f"{B}/2000-fra-2-stress-add-T-NSS/soap_test_100.xyz",
              "floor": 1.589,
              "models": {"v21": f"{B}/v21-SR-20260806/results/Na-NSS-v21-SR_stagetwo.model",
                         "v1": f"{B}/2000-fra-2-stress-add-T-NSS/results/Na-NSS-2000_stagetwo.model"}},
    "LPSCl": {"host": "/work/home/USER/Li-LPSCl/soap_selected_200_test.xyz",
              "floor": 1.337,
              "models": {"ft": "/work/home/USER/Li-LPSCl/train-numpt-15000/results/Li-LPSCl_stagetwo.model"}},
    "LLZO": {"host": "/work/home/USER/LLZO-SUN/multi-head-train/2000-fra/soap_test_200.xyz",
             "floor": 1.075,
             "models": {"ft": "/work/home/USER/LLZO-SUN/multi-head-train/2000-fra/results/LLZO-SUN_stagetwo.model"}},
}


def hunt(calc, host0, floor, tag, part_file):
    """Checkpoint every 20 starts into part_file so 1-hour malleable
    slices bank partial progress (the per-model file was too coarse)."""
    state = {"done": 0, "collapsed_dmin": 0, "big_fake_drops": 0,
             "worst_dmin_A": 9.9, "drops": []}
    if os.path.exists(part_file):
        state = json.load(open(part_file))
        print(f"  [{tag}] resume from {state['done']}/{NHUNT}", flush=True)
    # replay RNG stream deterministically up to the resume point
    rng = np.random.default_rng(20260830)
    seeds = [int(rng.integers(1e6)) for _ in range(NHUNT)]
    for k in range(state["done"], NHUNT):
        at = host0.copy()
        if k % 2 == 1:
            at.set_cell(at.cell * 0.97, scale_atoms=True)
        at.rattle(stdev=0.15, seed=seeds[k])
        at.calc = calc
        try:
            e0 = at.get_potential_energy() / len(at)
            opt = FIREOpt(at, maxstep=0.05, logfile=None)
            opt.run(fmax=0.3, steps=200)
            _, _, dd = neighbor_list("ijd", at, cutoff=3.0)
            dmin = float(dd.min()) if len(dd) else 3.0
            state["worst_dmin_A"] = min(state["worst_dmin_A"], dmin)
            de = at.get_potential_energy() / len(at) - e0
            if dmin < floor:
                state["collapsed_dmin"] += 1
            if de < -1.0:
                state["big_fake_drops"] += 1
                if len(state["drops"]) < 50:
                    state["drops"].append({"k": k, "dE": round(float(de), 3),
                                           "dmin": round(dmin, 3)})
        except Exception:
            state["big_fake_drops"] += 1
        state["done"] = k + 1
        if (k + 1) % 20 == 0 or k + 1 == NHUNT:
            json.dump(state, open(part_file, "w"))
            print(f"  [{tag}] {k+1}/{NHUNT} fake_drops="
                  f"{state['big_fake_drops']} "
                  f"worst={state['worst_dmin_A']:.2f}", flush=True)
    return {"n": NHUNT, "collapsed_dmin": state["collapsed_dmin"],
            "big_fake_drops": state["big_fake_drops"],
            "worst_dmin_A": round(state["worst_dmin_A"], 3), "floor": floor,
            "drops": state["drops"]}


for sname, cfg in SYSTEMS.items():
    host = read(cfg["host"], 0)
    rep = 1
    while len(host) * rep ** 3 < 300:
        rep += 1
    host0 = host.repeat((rep, rep, rep))
    models = dict(cfg["models"])
    models["mpa0"] = MPA0
    for mname, mpath in models.items():
        out = f"hunt100_{sname}_{mname}.json"
        if os.path.exists(out):
            print(f"[{sname}/{mname}] exists, skip", flush=True)
            continue
        print(f"[{sname}/{mname}] hunting n={NHUNT} "
              f"({len(host0)} atoms)", flush=True)
        try:
            calc = MACECalculator(model_paths=mpath, device="cuda",
                                  default_dtype="float64", enable_cueq=True)
            r = hunt(calc, host0, cfg["floor"], f"{sname}/{mname}",
                     f"hunt100_part_{sname}_{mname}.json")
            r["model"] = mpath
            json.dump(r, open(out, "w"), indent=1)
            print(f"[{sname}/{mname}] DONE fake_drops={r['big_fake_drops']}"
                  f"/{NHUNT} collapsed={r['collapsed_dmin']}", flush=True)
            del calc
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[{sname}/{mname}] FAILED: {exc}", flush=True)

print("HUNT100_ALL_DONE", flush=True)
