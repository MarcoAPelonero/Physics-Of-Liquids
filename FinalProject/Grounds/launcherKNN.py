from utils.importUtils import read_lammps_dump, compute_actual_positions, filter_atoms_by_radius
from utils.parallelQ import compute_Ql_knn_from_atoms
import torch
import os   

def main(n_neighbors: int = 6):
    """
    Compute local and global Q_l using a fixed number of nearest neighbours.
    """
    bounds, atoms = read_lammps_dump("data/last.dump")
    atoms = compute_actual_positions(bounds, atoms)
    atoms = filter_atoms_by_radius(atoms, 0.5)

    # Print how many atoms are left after filtering
    print(f"Number of atoms after filtering: {len(atoms)}")

    l_list = [4, 6, 8, 10, 12, 14]
    box    = torch.tensor(bounds, dtype=torch.float32)
    results = compute_Ql_knn_from_atoms(atoms, l_list, n_neighbors, box)

    # For each l, save a file in the knn_results directory
    output_dir = "knn_results"
    os.makedirs(output_dir, exist_ok=True)

    # The results should be a list of the singular q values for each atom, not the global or average ones
    for l, result in results.items():
        q_l_atom = result['atom_indices']
        output_file = os.path.join(output_dir, f"q_l_{l}.txt")  
        with open(output_file, 'w') as f:
            f.write(f"l = {l}\n")
            f.write("Index\tX\tY\tZ\tQ_l_atom\n")
            for atom_info in q_l_atom:
                idx = atom_info['index']
                x = atoms[idx]['x']
                y = atoms[idx]['y']
                z = atoms[idx]['z']
                f.write(f"{idx}\t{x:.6f}\t{y:.6f}\t{z:.6f}\t{atom_info['q_l_atom']:.6f}\n")
    print(f"Results saved in {output_dir} directory.")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute local and global Q_l using a fixed number of nearest neighbours.")
    parser.add_argument("--n_neighbors", type=int, default=8, help="Number of nearest neighbours to consider.")
    args = parser.parse_args()

    main(n_neighbors=args.n_neighbors)