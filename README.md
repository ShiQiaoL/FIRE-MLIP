# FIRE-MLIP

Code to reproduce *FIRE: Fine-Tuning and Stress-Testing Universal Interatomic Potentials for Solid–Solid Battery Interfaces* (Liu et al., arXiv:2601.17847).

FIRE = Fine-tuning with Integrated Replay and Efficiency: representative sampling (SOAP → PCA → clustering) of
MD candidates, replay-augmented fine-tuning of the MACE-MPA-0 foundation model, and a model-only stress test
before deployment (the FIRE-Audit toolkit). Data and models live on Hugging Face:

- datasets: https://huggingface.co/datasets/shiqiao123/FIRE-battery-interface-datasets
- models: https://huggingface.co/shiqiao123/FIRE-battery-interface-models


## Layout

| Folder | Content |
|---|---|
| `sampling/` | Representative-frame selection. `soap_pca_select.py` and `soap_pca_analysis.py` are the selection and analysis scripts; `convert_data.py` converts MTP `.cfg` to extended XYZ. |
| `training/` | The exact `run_train.sh` / Slurm scripts of every model in the paper: the six systems, the four Na/Na₃SbS₄ generations (FIRE, +SR, +SR′, +SR′+3B) and the learning-curve runs of Fig. 3 (FIRE, vanilla fine-tuning, random sampling, from-scratch training at 100–4000 frames). Hyperparameters are read directly from these files; Section S.6 of the SI summarises them. |
| `audit/` | FIRE-Audit toolkit: grouped errors, pair scans, hole hunt, short molecular dynamics, and the archived results and interface records. See `audit/README.md`. |
| `analysis/` | Figure and table scripts: `fig5_application.py` (Fig. 5), `kmod/` (DFT-anchored dimer curves of Fig. 5C), `revision/` (audit sweep, hole hunts, robustness grid, threshold sensitivity), `kdimer_LiClGaF3_recompute.py`. |
| `results/` | JSON outputs behind Fig. 5 and Sections S.5–S.6. |

## Environment

```
pip install -r requirements.txt
```

MACE 0.3.12 (0.3.13 for Li₃PS₄/Li₃B₁₁O₁₈), PyTorch, ASE ≥ 3.22, NumPy, SciPy, scikit-learn, dscribe (SOAP),
matplotlib. Training was done on NVIDIA A800 GPUs with `enable_cueq`; audits and dimer curves run on CPU.

## Reproducing the paper

1. **Sampling** (Fig. 3A–B): run `sampling/soap_pca_select.py` on an MD candidate pool; it writes `soap_train_*.xyz` / `soap_test_*.xyz`.
2. **Fine-tuning** (Figs. 2–3): download `mace-mpa-0-medium.model` from the MACE foundation-model release
   (SHA-256 `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`), place it next to the chosen
   `training/<system>/run_train.sh`, download that system's split from the dataset repository, and run the script.
3. **Stress test** (Fig. 5B): `python audit/audit/audit_g2x.py --model <model> --host <frame>` for the pair-scan gate and
   `python audit/audit/hunt100.py` for the hole hunt; criteria and thresholds are those archived in `audit/gates/`.
4. **DFT dimer anchors** (Fig. 5A, C): `analysis/kmod/gen_k.py` writes the VASP inputs, `collect_k.py` collects
   `kmod_curves.json`, `kdimer_model.py` evaluates the models on the same grid.
5. **Figure 5**: `python analysis/fig5_application.py --data results --out fig5`.

## Licence

Code: MIT (see `LICENSE`). Data and models are licensed separately in their repositories.
