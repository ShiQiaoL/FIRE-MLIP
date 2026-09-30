#!/usr/bin/env python
"""Package modules A + G (Fig 3a / Fig 2b data).

A: five-model predictions on the 90-dimer DFT reference set (curves vs DFT).
G: data-efficiency learning curves — evaluate the LLZO 100..8000-frame series
   (scratch + randselect) and the Li-LPSCl numpt series on their test sets.

Outputs: dimerA_results.json, geff_results.json (both plot-ready).
"""
import glob
import json
import os
import numpy as np
from collections import defaultdict
from ase.io import read
from mace.calculators import MACECalculator
import torch

B = "/work/home/USER/Na-NSS-mace/new-mace-Na-NSS-20250605/multi-head-train"
NA_MODELS = {
    "mpa0": f"{B}/2000-fra-2-stress-add-T-NSS/mace-mpa-0-medium.model",
    "v1": f"{B}/2000-fra-2-stress-add-T-NSS/results/Na-NSS-2000_stagetwo.model",
    "v2": f"{B}/v2-SR-20260805/results/Na-NSS-v2-SR_stagetwo.model",
    "v21": f"{B}/v21-SR-20260806/results/Na-NSS-v21-SR_stagetwo.model",
    "v22": f"{B}/v22-SR-20260807/results/Na-NSS-v22-SR_stagetwo.model",
}
DIMER_FILE = "/work/home/USER/Na-NSS-mace/mace-prep-20260801/stage1_dimer_diag.extxyz"


def ref_of(at):
    e = f = None
    if at.calc is not None:
        r = at.calc.results
        e = r.get("free_energy", r.get("energy"))
        f = r.get("forces")
    if e is None:
        for k in ("free_energy", "energy", "REF_energy", "dft_energy"):
            if k in at.info:
                e = at.info[k]
                break
    if f is None:
        for k in ("forces", "REF_forces", "dft_forces", "force"):
            if k in at.arrays:
                f = at.arrays[k]
                break
    return float(e), np.asarray(f, float)


def load_calc(path):
    return MACECalculator(model_paths=path, device="cuda",
                          default_dtype="float64", enable_cueq=True)


# ---------- Module A ----------
if os.path.exists("dimerA_results.json"):
    print("[A] dimerA_results.json exists, skipping module A", flush=True)
else:
    frames = read(DIMER_FILE, ":")
    print(f"[A] {len(frames)} dimer frames", flush=True)
    rows = []
    for at in frames:
        e, f = ref_of(at)
        rows.append({"pair": at.info.get("gen_pair"), "d": float(at.info.get("gen_d_target")),
                     "E_dft": e, "F_dft_x1": float(f[1, 0])})
    for name, path in NA_MODELS.items():
        try:
            calc = load_calc(path)
            for k, at in enumerate(frames):
                a = at.copy()
                a.calc = calc
                rows[k][f"E_{name}"] = float(a.get_potential_energy())
                rows[k][f"F_{name}_x1"] = float(a.get_forces()[1, 0])
            del calc
            torch.cuda.empty_cache()
            print(f"[A] {name} done", flush=True)
        except Exception as exc:
            print(f"[A] {name} FAILED: {exc}", flush=True)
    json.dump(rows, open("dimerA_results.json", "w"), indent=1)

# ---------- Module G ----------
import hashlib

LZ = "/work/home/USER/LLZO-SUN"
SERIES = [
    ("LLZO_scratch", f"{LZ}/scratch-train/*-fra"),
    ("LLZO_naive", f"{LZ}/naive-train/*-fra*"),
    ("LLZO_multihead", f"{LZ}/multi-head-train/*-fra*"),
    ("LLZO_randselect", f"{LZ}/randselect/*-randselect"),
    ("LLZO_seed100", f"{LZ}/randselect/seed-100-test/100-seed*"),
]
COMMON_TEST = f"{LZ}/scratch-train/8000-fra/test_800_frames.xyz"
LP_TEST = "/work/home/USER/Li-LPSCl/soap_selected_200_test.xyz"


def fhash(at):
    return hashlib.md5(np.round(at.positions, 5).tobytes()
                       + at.numbers.tobytes()).hexdigest()


def train_hashes(d):
    """Hash set of this dir's training frames (for common-test overlap removal)."""
    tr = sorted(glob.glob(f"{d}/*train*.xyz"))
    if not tr:
        return None, None
    hs = set()
    for at in read(tr[0], ":"):
        hs.add(fhash(at))
    return hs, tr[0]


def eval_frames(calc, frames):
    de, df = [], []
    for at in frames:
        e0, f0 = ref_of(at)
        n = len(at)
        a = at.copy()
        a.calc = calc
        de.append((a.get_potential_energy() - e0) / n * 1000)
        df.append(((a.get_forces() - f0) * 1000).ravel())
    de = np.array(de)
    df = np.concatenate(df)
    return {"n_frames": int(de.size),
            "E_rmse_meV_atom": round(float(np.sqrt((de ** 2).mean())), 3),
            "F_rmse_meV_A": round(float(np.sqrt((df ** 2).mean())), 2)}


common_pool = read(COMMON_TEST, ":")
common_h = [fhash(a) for a in common_pool]
print(f"[G] common LLZO pool: {len(common_pool)} frames", flush=True)

geff = []
done = set()
if os.path.exists("geff_results.json"):
    old = json.load(open("geff_results.json"))
    geff = [r for r in old if "error" not in r]
    done = {(r["series"], r["tag"]) for r in geff}
    print(f"[G] resume: {len(done)} entries already done", flush=True)
for series, pat in SERIES:
    for d in sorted(glob.glob(pat)):
        if not os.path.isdir(d) or os.path.basename(d) == "seed-100-test":
            continue
        tag = os.path.basename(d)
        if (series, tag) in done:
            continue
        models = sorted(glob.glob(f"{d}/results/*_stagetwo.model"))
        models = [m for m in models if "run-" not in os.path.basename(m)] or models
        if not models:
            geff.append({"series": series, "tag": tag, "error": "no model"})
            continue
        row = {"series": series, "tag": tag, "model": models[0]}
        try:
            hs, trf = train_hashes(d)
            if hs is None:
                use, row["overlap_note"] = common_pool, "no train file found; no exclusion"
            else:
                use = [a for a, h in zip(common_pool, common_h) if h not in hs]
                row["train_file"] = trf
                row["n_overlap_excluded"] = len(common_pool) - len(use)
            calc = load_calc(models[0])
            row.update(eval_frames(calc, use))
            own = sorted(glob.glob(f"{d}/*test*.xyz"))
            if own and os.path.realpath(own[0]) != os.path.realpath(COMMON_TEST):
                r_own = eval_frames(calc, read(own[0], ":"))
                row["own_test"] = own[0]
                row["own_E_rmse_meV_atom"] = r_own["E_rmse_meV_atom"]
                row["own_F_rmse_meV_A"] = r_own["F_rmse_meV_A"]
            del calc
            torch.cuda.empty_cache()
            print(f"[G] {series}/{tag}: {row.get('E_rmse_meV_atom')} meV/at, "
                  f"{row.get('F_rmse_meV_A')} meV/A "
                  f"(excl {row.get('n_overlap_excluded')})", flush=True)
        except Exception as exc:
            row["error"] = str(exc)
            print(f"[G] {series}/{tag} FAILED: {exc}", flush=True)
        geff.append(row)
        json.dump(geff, open("geff_results.json", "w"), indent=1)

# LPSCl numpt series (dedicated 200-frame test set at parent level)
lp_pool = read(LP_TEST, ":")
lp_h = [fhash(a) for a in lp_pool]
for d in sorted(glob.glob("/work/home/USER/Li-LPSCl/train-numpt-*")):
    tag = os.path.basename(d)
    if "yswang" in tag or ("LPSCl_numpt", tag) in done:
        continue
    models = sorted(glob.glob(f"{d}/results/*_stagetwo.model"))
    models = [m for m in models if "run-" not in os.path.basename(m)] or models
    if not models:
        geff.append({"series": "LPSCl_numpt", "tag": tag, "error": "no model"})
        continue
    row = {"series": "LPSCl_numpt", "tag": tag, "model": models[0], "test": LP_TEST}
    try:
        hs, trf = train_hashes(d)
        if hs is None:
            use = lp_pool
        else:
            use = [a for a, h in zip(lp_pool, lp_h) if h not in hs]
            row["train_file"] = trf
            row["n_overlap_excluded"] = len(lp_pool) - len(use)
        calc = load_calc(models[0])
        row.update(eval_frames(calc, use))
        del calc
        torch.cuda.empty_cache()
        print(f"[G] LPSCl/{tag}: {row.get('E_rmse_meV_atom')} meV/at, "
              f"{row.get('F_rmse_meV_A')} meV/A", flush=True)
    except Exception as exc:
        row["error"] = str(exc)
        print(f"[G] LPSCl/{tag} FAILED: {exc}", flush=True)
    geff.append(row)
    json.dump(geff, open("geff_results.json", "w"), indent=1)

print("ALL DONE", flush=True)
