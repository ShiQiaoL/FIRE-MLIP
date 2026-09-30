from ase import Atoms
from ase.io import Trajectory
import numpy as np

def parse_quip_cfg_to_ase(cfg_file):
    with open(cfg_file, 'r') as f:
        lines = f.readlines()

    i = 0
    atoms_list = []

    while i < len(lines):
        if lines[i].strip() == "BEGIN_CFG":
            i += 1
            num_atoms = 0
            positions = []
            atom_types = []
            forces = []
            magmoms = []
            mag_forces = []
            energy = None
            cell = []

            while lines[i].strip() != "END_CFG":
                line = lines[i].strip()

                if line == "Size":
                    i += 1
                    num_atoms = int(lines[i].strip())

                elif line == "Supercell":
                    i += 1
                    cell = []
                    for _ in range(3):
                        cell.append([float(x) for x in lines[i].strip().split()])
                        i += 1

                elif line.startswith("AtomData:"):
                    i += 1
                    for _ in range(num_atoms):
                        tokens = lines[i].split()
                        atom_types.append(int(tokens[1]))
                        positions.append([float(x) for x in tokens[2:5]])
                        forces.append([float(x) for x in tokens[5:8]])
                        magmoms.append([float(x) for x in tokens[8:11]])
                        mag_forces.append([float(x) for x in tokens[11:14]])
                        i += 1

                elif line.startswith("Energy"):
                    i += 1
                    energy = float(lines[i].strip())
                elif line.startswith("PlusStress:"):
                    i += 1
                    stress_components = list(map(float, lines[i].strip().split()))
                    stress = np.array([
                        [stress_components[0], stress_components[5], stress_components[4]],
                        [stress_components[5], stress_components[1], stress_components[3]],
                        [stress_components[4], stress_components[3], stress_components[2]]
                    ])
                else:
                    i += 1

            type_map = {0: 'Li', 1: 'P', 2: 'S', 3: 'B', 4: 'O'}   
            symbols = [type_map[t] for t in atom_types]

            atoms = Atoms(
                symbols=symbols,
                positions=np.array(positions),
                cell=np.array(cell),
                pbc=True
            )
            atoms.info["dft_energy"] = energy
            atoms.arrays["dft_forces"] = np.array(forces)
            atoms.arrays["dft_magmom"] = np.array(magmoms)
            atoms.arrays["REF_magforces"] = np.array(mag_forces)
            atoms.info["dft_stress"] = stress / -atoms.get_volume()
            atoms_list.append(atoms)
        i += 1

    return atoms_list

# Usage
atoms_list = parse_quip_cfg_to_ase("training_set.cfg")
import ase.io
ase.io.write("training_set_REF_magforces.xyz", atoms_list)