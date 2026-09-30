# FIRE-Audit

Stress-test toolkit and archived results accompanying *FIRE: Fine-Tuning and Stress-Testing Universal Interatomic Potentials for Solid–Solid Battery Interfaces*.
All paths are relative to this directory.

## Model names

The scripts and result files use the working names of the models; the paper uses the names on the right.

| Name in files | Name in the paper |
|---|---|
| `mpa0` | MACE-MPA-0 (foundation model) |
| `v1` | FIRE |
| `v2` | FIRE+SR |
| `v21`, `v2.1` | FIRE+SR' |
| `v22`, `v2.2` | FIRE+SR'+3B |
| `ft` | fine-tuned model of the respective system |

## Layout

```
audit/                 Stress-test scripts for Na/Na3SbS4 (ASE + MACE)
  eval_g1.py           grouped energy/force RMSE
  audit_g2.py          pair scans (no spurious minimum, repulsive force,
                       wall height) and randomized hole hunt
  g3_pilot.py          multi-seed short molecular dynamics
  g3_diag.py           diagnostic variant with a rolling frame buffer
audit_v2/              Six-system version: pair scans, hole hunt (n = 100),
                       robustness-grid driver, threshold sensitivity
gates/                 Archived reports of the five Na/Na3SbS4 models
  g1_report.json       grouped RMSE
  g2_report.json       pair-scan verdicts: 16/16, 10/16, 13/16, 16/16, 16/16
  g2_scans.npz         raw pair-scan curves
  g3*_report.json      short-MD outcomes, with the failing frames
analysis/              Machine-readable results used in Fig. 5 and the SI
data/
  dimer_ref/           90 vacuum dimers with DFT labels
  repair_dft/          short-range repair sets with DFT labels
deployment_postmortem/ Frames and records of the 2712-atom interface runs
```

## DFT settings

PBE, PAW (Na_pv, S, Sb), ENCUT 520 eV, PREC Accurate, ISMEAR 0 with SIGMA 0.1,
Gamma-centred grid, EDIFF 1e-6 eV; the energy label is the electronic free energy.

## Not included

- The foundation model `mace-mpa-0-medium`: obtain it from the MACE release.
- POTCAR files (VASP licence).
