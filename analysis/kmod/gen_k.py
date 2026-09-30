#!/usr/bin/env python3
"""Trimmed module K: DFT dimer anchors for the audit's robust-FAIL and
verdict-named pairs (thresh_sens.json informed).

7 pairs x 14 distances = 98 VASP singles, 16 A cubic box, gamma-only,
stage1 fingerprint (PREC Accurate / ENCUT 520 / EDIFF 1e-6 / ISMEAR 0
SIGMA 0.1) + ISPIN 2 (open-shell dimers) + LREAL .FALSE. (2-atom cell)
+ safe mixing. Na-NSS pairs excluded: stage1_dimer_diag already carries
DFT labels for them.

Run on eshell111 in 08_kmod/. Then bash submit_all.sh.
"""
import os

PAW = "/work/home/USER/pseudopotentials/PAW_PBE"
PAIRS = [
    ("C-O", [("C", "C"), ("O", "O")]),
    ("B-O", [("B", "B"), ("O", "O")]),
    ("Ga-Ga", [("Ga", "Ga_d")]),
    ("Li-Li", [("Li", "Li_sv")]),
    ("S-S", [("S", "S")]),
    ("F-Li", [("F", "F"), ("Li", "Li_sv")]),
    ("Li-O", [("Li", "Li_sv"), ("O", "O")]),
]
DISTS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 1.9, 2.2, 2.5, 2.8, 3.2, 3.6]

INCAR = """SYSTEM = {name}
PREC = Accurate
ENCUT = 520
EDIFF = 1E-6
ALGO = Normal
NELM = 200
ISMEAR = 0
SIGMA = 0.1
ISYM = 0
ISPIN = 2
MAGMOM = 2*1.0
LREAL = .FALSE.
AMIX = 0.2
BMIX = 0.0001
NSW = 0
IBRION = -1
LWAVE = .FALSE.
LCHARG = .FALSE.
KSPACING = 1.2
KGAMMA = .TRUE.
NPAR = 2
"""

SLURM = """#!/bin/bash
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -p xahcnormal
#SBATCH -o vasp-%j.out
#SBATCH -e vasp-%j.err

module purge
module load vasp-6.4.2-intelmpi2017_ioptcell

export MKL_DEBUG_CPU_TYPE=5
export MKL_CBWR=AVX2
export I_MPI_PIN_DOMAIN=numa

srun --mpi=pmi2 vasp_gam
"""

subs = []
for pname, species in PAIRS:
    if len(species) == 1:
        elems = [species[0]] * 2
        uniq = [species[0]]
        counts = [2]
    else:
        elems = species
        uniq = species
        counts = [1, 1]
    for t in DISTS:
        tag = f"k_{pname.replace('-', '')}_d{t:.2f}".replace(".", "p")
        os.makedirs(tag, exist_ok=True)
        lines = [tag, "1.0", "16.0 0.0 0.0", "0.0 16.0 0.0", "0.0 0.0 16.0",
                 " ".join(u[0] for u in uniq),
                 " ".join(str(c) for c in counts), "Cartesian",
                 "0.0 0.0 0.0", f"{t:.4f} 0.0 0.0"]
        open(f"{tag}/POSCAR", "w").write("\n".join(lines) + "\n")
        open(f"{tag}/INCAR", "w").write(INCAR.format(name=tag))
        open(f"{tag}/vasp.slurm", "w").write(SLURM.format(name=tag))
        with open(f"{tag}/POTCAR", "wb") as fo:
            for u in uniq:
                fo.write(open(f"{PAW}/{u[1]}/POTCAR", "rb").read())
        subs.append(tag)

open("submit_all.sh", "w").write(
    "#!/bin/bash\nfor d in " + " ".join(subs)
    + "; do (cd $d && sbatch vasp.slurm); done\n")
print(f"generated {len(subs)} dirs; run: bash submit_all.sh")
