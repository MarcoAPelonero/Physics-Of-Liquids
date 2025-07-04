#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.importUtils import read_lammps_dump, compute_actual_positions, filter_atoms_by_radius
from utils.parallelQ import compute_rdf

sns.set_style("whitegrid")
def main(filename):
    box, atoms = read_lammps_dump(filename)
    atoms = filter_atoms_by_radius(atoms, 0.5)  
    new_atoms = compute_actual_positions(box, atoms)
    positions = np.array([[atom['x'], atom['y'], atom['z']] for atom in new_atoms])

    r, g = compute_rdf(positions, box, dr=0.02)

    plt.figure(figsize=(6,4))
    plt.plot(r, g, '-o', markersize=4, color='purple', linewidth=1.5)
    plt.fill_between(r, g, color='purple', alpha=0.1)
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0.8, 5)
    plt.savefig("radial_distribution_function.png", dpi=300)
    plt.show()
    sns.despine()

if __name__ == "__main__":
    main("data/last.dump")