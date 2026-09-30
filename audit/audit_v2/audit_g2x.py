#!/usr/bin/env python
"""B-lite: five-system G2 audit sweep (generalized, self-calibrating).

Differences vs the Na-NSS audit_g2.py (criteria v1.1):
  - pair list and d_data thresholds are DERIVED from each system's training
    set (d_data = closest approach present in the data = support boundary),
    not hand-coded chemistry tables;
  - embedded-environment scans only (bulk test frame host); vacuum dimers
    are covered by the Na-NSS deep case (module A) and were the source of
    the molecular-minimum false positives;
  - wall grading rule reproducing v1.1: both-alkali 3 eV / contains-alkali
    5 eV / else 8 eV;
  - hole hunt: 20 rattled(0.15)+3%-compressed starts on a >=300-atom
    supercell, capped FIRE; collapse when dmin < min(d_data) - 0.3.

Per system: foundation (mpa0) vs fine-tuned model (Na-NSS also v1).
Outputs g2x_report.json + g2x_scans.npz.
"""
import glob
import json
import numpy as np
import torch
from ase.io import read
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE as FIREOpt
from mace.calculators import MACECalculator

MPA0 = ("/work/home/USER/Na-NSS-mace/new-mace-Na-NSS-20250605/"
        "multi-head-train/2000-fra-2-stress-add-T-NSS/mace-mpa-0-medium.model")
B = "/work/home/USER/Na-NSS-mace/new-mace-Na-NSS-20250605/multi-head-train"

SYSTEMS = {
    "NaNSS": {
        "models": {"v21": f"{B}/v21-SR-20260806/results/Na-NSS-v21-SR_stagetwo.model",
                   "v1": f"{B}/2000-fra-2-stress-add-T-NSS/results/Na-NSS-2000_stagetwo.model"},
        "train": f"{B}/2000-fra-2-stress-add-T-NSS/soap_train_1800.xyz",
        "host": f"{B}/2000-fra-2-stress-add-T-NSS/soap_test_100.xyz",
    },
    "LPSCl": {
        "models": {"ft": "/work/home/USER/Li-LPSCl/train-numpt-15000/results/Li-LPSCl_stagetwo.model"},
        "train": "/work/home/USER/Li-LPSCl/train-numpt-15000/*train*.xyz",
        "host": "/work/home/USER/Li-LPSCl/soap_selected_200_test.xyz",
    },
    "LLZO": {
        "models": {"ft": "/work/home/USER/LLZO-SUN/multi-head-train/2000-fra/results/LLZO-SUN_stagetwo.model"},
        "train": "/work/home/USER/LLZO-SUN/multi-head-train/2000-fra/soap_train_1800.xyz",
        "host": "/work/home/USER/LLZO-SUN/multi-head-train/2000-fra/soap_test_200.xyz",
    },
    "LiClGaF3": {
        "models": {"ft": "/work/home/USER/LiCl-GaF3/remove-errstructure/results1/LiCl-GaF3_stagetwo.model"},
        # train2.xyz is the ACTUAL training file (per run_train.sh); train.xyz
        # is the pre-cleaning file with error structures (pairs down to 0.3 A)
        "train": "/work/home/USER/LiCl-GaF3/remove-errstructure/train2.xyz",
        "host": "/work/home/USER/LiCl-GaF3/remove-errstructure/test.xyz",
    },
    "Li2CO3LiF": {
        "models": {"ft": "/work/home/USER/Li2CO3-LiF/results1/Li2CO3-LiF_stagetwo.model"},
        "train": "/work/home/USER/Li2CO3-LiF/mace_train.xyz",
        "host": "/work/home/USER/Li2CO3-LiF/mace_test.xyz",
    },
    "LPSLBO": {
        # filename is a copy-residue from the Li2CO3 script; log-verified
        # atomic numbers [3,5,8,15,16] = Li,B,O,P,S -> Li3PS4-Li3B11O18
        "models": {"ft": "/work/home/USER/LPS-LBO/results1/Li2CO3-LiF_stagetwo.model"},
        "train": "/work/home/USER/LPS-LBO/soap_train_1800.xyz",
        "host": "/work/home/USER/LPS-LBO/soap_test_200.xyz",
    },
}

ALKALI = {"Li", "Na", "K"}
MIN_PROMINENCE = 0.05
GRID = np.round(np.arange(0.8, 4.0001, 0.05), 2)
MAX_TRAIN_FRAMES = 300
NHUNT = 20


def wall_min(A, B_):
    if A in ALKALI and B_ in ALKALI:
        return 3.0
    if A in ALKALI or B_ in ALKALI:
        return 5.0
    return 8.0


def derive_d_data(train_glob):
    """Per pair: dmin = support boundary (closest approach in the data);
    d_eq = p10 of observed distances (lower edge of the thermal first shell,
    robust vs isolated compressed frames) — the wall/force reference point."""
    files = sorted(glob.glob(train_glob))
    frames = read(files[0], ":")
    if len(frames) > MAX_TRAIN_FRAMES:
        idx = np.linspace(0, len(frames) - 1, MAX_TRAIN_FRAMES).astype(int)
        frames = [frames[k] for k in idx]
    acc = {}
    for at in frames:
        syms = np.array(at.get_chemical_symbols())
        i, j, d = neighbor_list("ijd", at, cutoff=4.0)
        for k in range(len(d)):
            key = tuple(sorted((syms[i[k]], syms[j[k]])))
            acc.setdefault(key, []).append(float(d[k]))
    dd = {k: {"dmin": round(float(np.min(v)), 3),
              "d_eq": round(float(np.percentile(v, 10)), 3)}
          for k, v in acc.items()}
    return dd, files[0], len(frames)


def mic_pair(at, i, j):
    v, d = find_mic((at.positions[j] - at.positions[i]).reshape(1, 3),
                    at.cell, at.pbc)
    return v[0], d[0]


def squeeze(host, i, j, target):
    at = host.copy()
    v, dist = mic_pair(at, i, j)
    u = v / dist
    s = (dist - target) / 2.0
    at.positions[i] += u * s
    at.positions[j] -= u * s
    return at


def pick_pair(host, A, B_):
    syms = np.array(host.get_chemical_symbols())
    i, j, d = neighbor_list("ijd", host, cutoff=5.0)
    for k in np.argsort(d):
        if syms[i[k]] == A and syms[j[k]] == B_:
            return int(i[k]), int(j[k])
    return None


def audit_scan(d, E, Fp, d_bound, d_eq, wmin):
    res = {"d_min_data": d_bound, "d_eq_p10": d_eq}
    lo = d < min(d_eq - 0.3, d_bound)
    mins = []
    for k in range(1, len(d) - 1):
        if (lo[k] and E[k] < E[k - 1] - MIN_PROMINENCE
                and E[k] < E[k + 1] - MIN_PROMINENCE):
            mins.append(float(d[k]))
    res["local_minima_below_data"] = mins
    comp = d < 0.85 * d_eq
    bad = d[comp][np.asarray(Fp)[comp] < 0.0]
    res["attractive_points_compressed"] = [float(x) for x in bad]
    kref = int(np.argmin(np.abs(d - min(d_eq, 3.95))))
    k12 = int(np.argmin(np.abs(d - max(1.2, d[0]))))
    res["wall_height_eV"] = float(E[k12] - E[kref])
    res["wall_min_eV"] = wmin
    res["pass"] = (not mins) and (len(bad) == 0) and res["wall_height_eV"] >= wmin
    return res


import os

report = {}
scans = {}
prev_report = {}
if os.path.exists("g2x_report.json"):
    prev_report = json.load(open("g2x_report.json"))
    print(f"[resume] prior report with {list(prev_report)} loaded", flush=True)
if os.path.exists("g2x_scans.npz"):
    z = np.load("g2x_scans.npz")
    scans = {k: z[k] for k in z.files if k != "d"}
    print(f"[resume] {len(scans)} prior scan curves loaded", flush=True)

for sname, cfg in SYSTEMS.items():
    d_data, train_file, ntr = derive_d_data(cfg["train"])
    host = read(cfg["host"], 0)
    rep_h = 1
    while len(host) * rep_h ** 3 < 300:
        rep_h += 1
    hunt_host0 = host.repeat((rep_h, rep_h, rep_h))
    dmin_floor = min(v["dmin"] for v in d_data.values()) - 0.3
    srep = {"train_file": train_file, "n_train_frames_scanned": ntr,
            "d_data": {f"{a}-{b}": v for (a, b), v in d_data.items()},
            "dmin_floor": round(dmin_floor, 3), "models": {}}
    models = dict(cfg["models"])
    models["mpa0"] = MPA0
    prev_models = (prev_report.get(sname) or {}).get("models", {})
    for mname, mpath in models.items():
        pm = prev_models.get(mname)
        if pm and "summary" in pm:
            srep["models"][mname] = pm
            print(f"[{sname}/{mname}] reused from prior run", flush=True)
            continue
        try:
            calc = MACECalculator(model_paths=mpath, device="cuda",
                                  default_dtype="float64", enable_cueq=True)
        except Exception as exc:
            srep["models"][mname] = {"error": f"load: {exc}"}
            continue
        mrep = {"model": mpath, "scans": {}}
        for (A, B_), dv in sorted(d_data.items()):
            pr = pick_pair(host, A, B_)
            if pr is None:
                continue
            i, j = pr
            es, fs = [], []
            try:
                for t in GRID:
                    at = squeeze(host, i, j, float(t))
                    at.calc = calc
                    es.append(at.get_potential_energy())
                    f = at.get_forces()
                    u, _ = mic_pair(at, i, j)
                    u = u / np.linalg.norm(u)
                    fs.append(float(np.dot(f[j], u)))
            except Exception as exc:
                mrep["scans"][f"{A}-{B_}"] = {"error": str(exc)}
                continue
            E, Fp = np.array(es), np.array(fs)
            scans[f"{sname}:{mname}:{A}-{B_}:E"] = E
            scans[f"{sname}:{mname}:{A}-{B_}:F"] = Fp
            mrep["scans"][f"{A}-{B_}"] = audit_scan(
                GRID, E, Fp, dv["dmin"], dv["d_eq"], wall_min(A, B_))
        rng = np.random.default_rng(20260819)
        n_bad_dmin, n_bad_dE, worst = 0, 0, 9.9
        for k in range(NHUNT):
            at = hunt_host0.copy()
            if k % 2 == 1:
                at.set_cell(at.cell * 0.97, scale_atoms=True)
            at.rattle(stdev=0.15, seed=int(rng.integers(1e6)))
            at.calc = calc
            try:
                e0 = at.get_potential_energy() / len(at)
                opt = FIREOpt(at, maxstep=0.05, logfile=None)
                opt.run(fmax=0.3, steps=200)
                _, _, dd_ = neighbor_list("ijd", at, cutoff=3.0)
                dmin = float(dd_.min()) if len(dd_) else 3.0
                worst = min(worst, dmin)
                de = at.get_potential_energy() / len(at) - e0
                if dmin < dmin_floor:
                    n_bad_dmin += 1
                if de < -1.0:
                    n_bad_dE += 1
            except Exception:
                n_bad_dE += 1
        mrep["hole_hunt"] = {"n": NHUNT, "collapsed_dmin": n_bad_dmin,
                             "big_fake_drops": n_bad_dE,
                             "worst_dmin_A": round(worst, 3),
                             "pass": n_bad_dmin == 0 and n_bad_dE == 0}
        ok = [v for v in mrep["scans"].values() if "pass" in v]
        npass = sum(1 for v in ok if v["pass"])
        mrep["summary"] = {"scans_pass": f"{npass}/{len(ok)}",
                           "hole_hunt_pass": mrep["hole_hunt"]["pass"]}
        srep["models"][mname] = mrep
        print(f"[{sname}/{mname}] scans {npass}/{len(ok)}; "
              f"hunt pass={mrep['hole_hunt']['pass']} "
              f"worst_dmin={worst:.2f}", flush=True)
        del calc
        torch.cuda.empty_cache()
        np.savez_compressed("g2x_scans.npz", d=GRID, **scans)
        json.dump(report | {sname: srep}, open("g2x_report.json", "w"), indent=1)
    report[sname] = srep

np.savez_compressed("g2x_scans.npz", d=GRID, **scans)
json.dump(report, open("g2x_report.json", "w"), indent=1)
print("G2X_DONE", flush=True)
