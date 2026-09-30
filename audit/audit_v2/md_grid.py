#!/usr/bin/env python
"""Module C: robustness grid — pure-model bulk NVT MD (no guards of any kind).

One invocation = one system (env SYS) x 3 temperatures x 4 seeds x 50 ps.
Sentinel is MONITORING-ONLY: it records warning events (dmin/T excursions)
and never modifies the dynamics. Survival is judged post hoc.

Per run outputs: md_{sys}_{T}K_s{seed}.json (+ traj subsample .extxyz).
"""
import json
import os
import sys
import time
import numpy as np
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from mace.calculators import MACECalculator

SYSTEMS = {
    "NaNSS": {
        "model": "/work/home/USER/Na-NSS-mace/new-mace-Na-NSS-20250605/multi-head-train/v21-SR-20260806/results/Na-NSS-v21-SR_stagetwo.model",
        "struct": "/work/home/USER/Na-NSS-mace/new-mace-Na-NSS-20250605/multi-head-train/2000-fra-2-stress-add-T-NSS/soap_test_100.xyz",
    },
    "LPSCl": {
        "model": "/work/home/USER/Li-LPSCl/train-numpt-15000/results/Li-LPSCl_stagetwo.model",
        "struct": "/work/home/USER/Li-LPSCl/soap_selected_200_test.xyz",
    },
    "LLZO": {
        "model": "/work/home/USER/LLZO-SUN/multi-head-train/2000-fra/results/LLZO-SUN_stagetwo.model",
        "struct": "/work/home/USER/LLZO-SUN/multi-head-train/2000-fra/soap_test_200.xyz",
    },
    "LiClGaF3": {
        "model": "/work/home/USER/LiCl-GaF3/remove-errstructure/results1/LiCl-GaF3_stagetwo.model",
        "struct": "/work/home/USER/LiCl-GaF3/remove-errstructure/test.xyz",
    },
    "Li2CO3LiF": {
        "model": "/work/home/USER/Li2CO3-LiF/results1/Li2CO3-LiF_stagetwo.model",
        "struct": "/work/home/USER/Li2CO3-LiF/mace_test.xyz",
    },
    "LPSLBO": {
        # misnamed file, log-verified Li/B/O/P/S (Li3PS4-Li3B11O18)
        "model": "/work/home/USER/LPS-LBO/results1/Li2CO3-LiF_stagetwo.model",
        "struct": "/work/home/USER/LPS-LBO/soap_test_200.xyz",
    },
}

TEMPS = [300, 600, 900]
SEEDS = [1, 2, 3, 4]
DT_FS = 2.0
NSTEPS = 25000          # 50 ps
CHECK_EVERY = 50
SAVE_EVERY = 500
TARGET_ATOMS = 300      # supercell up to at least this size

WARN_DMIN = 1.20        # record-only thresholds
HARD_DMIN = 0.80        # post-hoc death criterion
WARN_TFAC = 3.0

sys_name = os.environ.get("SYS") or sys.argv[1]
cfg = SYSTEMS[sys_name]

base = read(cfg["struct"], 0)
cell_len = np.linalg.norm(base.cell.array, axis=1)
best = None
for na in range(1, 4):
    for nb in range(1, 4):
        for nc in range(1, 4):
            n = len(base) * na * nb * nc
            if n < TARGET_ATOMS:
                continue
            short = min(cell_len * (na, nb, nc))
            if best is None or (n, -short) < (best[0], -best[1]):
                best = (n, short, (na, nb, nc))
rep = best[2] if best else (1, 1, 1)
atoms0 = base.repeat(rep)
print(f"[{sys_name}] base {len(base)} atoms x {rep} -> {len(atoms0)}", flush=True)

calc = MACECalculator(model_paths=cfg["model"], device="cuda",
                      default_dtype="float64", enable_cueq=True)


def dmin_of(at):
    d = at.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


for T in TEMPS:
    for seed in SEEDS:
        tag = f"md_{sys_name}_{T}K_s{seed}"
        if os.path.exists(f"{tag}.json"):
            print(f"[{tag}] exists, skip", flush=True)
            continue
        at = atoms0.copy()
        at.calc = calc
        rng = np.random.RandomState(seed)
        MaxwellBoltzmannDistribution(at, temperature_K=T, rng=rng)
        dyn = Langevin(at, DT_FS * units.fs, temperature_K=T,
                       friction=0.02, rng=rng)
        events, traj = [], []
        state = {"system": sys_name, "model": cfg["model"], "struct": cfg["struct"],
                 "n_atoms": len(at), "T": T, "seed": seed, "dt_fs": DT_FS,
                 "nsteps_target": NSTEPS}
        died = None
        max_cw = 0.0
        slow_streak = 0
        env_abort = False
        t_ch = time.time()
        try:
            for chunk in range(NSTEPS // CHECK_EVERY):
                dyn.run(CHECK_EVERY)
                cw = time.time() - t_ch
                t_ch = time.time()
                max_cw = max(max_cw, cw)
                step = (chunk + 1) * CHECK_EVERY
                ep = at.get_potential_energy()
                tk = at.get_temperature()
                dm = dmin_of(at)
                # wall-clock sentinel, PHYSICS-GATED: a dense-collapse blob
                # slows chunks AND crushes dmin; slow chunks with healthy
                # dmin are a contended node, not physics (LLZO-900K lesson)
                if cw > 90.0:
                    if dm < WARN_DMIN:
                        died = {"step": step, "reason": f"slowdown {cw:.0f}s/"
                                f"{CHECK_EVERY} steps + dmin {dm:.2f} "
                                "(dense collapse)",
                                "dmin": round(dm, 3), "T_inst": round(tk, 1)}
                        write(f"{tag}_died.extxyz", [at])
                        break
                    slow_streak += 1
                    if slow_streak >= 20:
                        env_abort = True
                        break
                else:
                    slow_streak = 0
                if not np.isfinite(ep):
                    died = {"step": step, "reason": "non-finite energy"}
                    break
                if dm < WARN_DMIN or tk > WARN_TFAC * T:
                    events.append({"step": step, "dmin": round(dm, 3),
                                   "T_inst": round(tk, 1)})
                if dm < HARD_DMIN:
                    died = {"step": step, "reason": f"dmin {dm:.2f} < {HARD_DMIN}"}
                    break
                if step % SAVE_EVERY == 0:
                    fr = at.copy()
                    fr.info.update(step=step, T_inst=round(tk, 1),
                                   E_pot=round(float(ep), 6))
                    traj.append(fr)
        except Exception as exc:
            died = {"step": "unknown", "reason": f"exception: {exc}"}
        if env_abort:
            # contended node: no verdict, no json — resume re-runs this tag
            print(f"[{tag}] NODE_SLOW abort at step {step} "
                  f"(dmin {dm:.2f} healthy); exiting for clean rerun", flush=True)
            sys.exit(3)
        state["survived"] = died is None
        state["died"] = died
        state["max_chunk_wall_s"] = round(max_cw, 1)
        state["n_warn_events"] = len(events)
        state["warn_events"] = events[:200]
        if traj:
            write(f"{tag}.extxyz", traj)
        json.dump(state, open(f"{tag}.json", "w"), indent=1)
        print(f"[{tag}] survived={state['survived']} warns={len(events)} "
              f"died={died}", flush=True)

print(f"{sys_name}_GRID_DONE", flush=True)
