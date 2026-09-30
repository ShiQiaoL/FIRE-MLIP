# FIRE-Audit contents

| Path | Content |
|---|---|
| `audit/` | Stress-test scripts for Na/Na3SbS4 |
| `audit_v2/` | Six-system stress test, hole hunt (n = 100), robustness-grid driver, threshold sensitivity |
| `gates/` | Reports, scan curves and evidence frames of the five Na/Na3SbS4 models |
| `analysis/` | Results behind Fig. 5 and Supplementary Sections S.5-S.6 |
| `data/dimer_ref/` | 90 vacuum dimers with DFT labels |
| `data/repair_dft/` | Short-range repair sets with DFT labels |
| `deployment_postmortem/` | Records and frames of the 2712-atom interface runs |

## Notes for users

1. Some model files on the training cluster carry names that do not match their content; for example the
   Li3PS4/Li3B11O18 model was saved as `Li2CO3-LiF_stagetwo.model`. The identity of every model was verified
   from the atomic numbers in its training log, and the released model files are named by system.
2. The LiCl/GaF3 entries of `analysis/g2x_report.json`, `hunt100_LiClGaF3_*.json` and the LiCl/GaF3 runs of the
   robustness grid were computed with an earlier 82-frame checkpoint of that system. They are kept for
   traceability and are not used in the paper. The LiCl/GaF3 dimer curves of Fig. 5C use the model of Fig. 2
   (`results/kdimer_model.json`; the earlier values are in `results/kdimer_model_results1.json`).
3. A global minimum-distance threshold of 1.2 A over-counts the vibrations of covalent bonds such as C-O in
   carbonate (`analysis/attrib_results.json`); stress tests have to be evaluated pair by pair.
