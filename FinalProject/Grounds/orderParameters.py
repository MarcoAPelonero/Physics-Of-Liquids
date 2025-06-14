import torch
from utils.importUtils import read_lammps_dump, compute_actual_positions, filter_atoms_by_radius
from utils.parallelQ import compute_Ql_from_atoms
import argparse

def main(cutoff_radius=None):
    # Read and prepare data
    bounds, atoms = read_lammps_dump("data/last.dump")
    atoms = compute_actual_positions(bounds, atoms)

    # If no cutoff passed, use legacy default of 0.5
    if cutoff_radius is None:
        cutoff_radius = 1.4
        legacy = True
    else:
        legacy = False

    # Filter atoms by radius
    atoms = filter_atoms_by_radius(atoms, 0.5)

    # Parameters for Ql computation
    l_list = [4, 6, 8, 10, 12]
    cutoff_radius = float(cutoff_radius)
    box = torch.tensor(bounds, dtype=torch.float32)

    # Compute Ql and (optionally) global Q
    results = compute_Ql_from_atoms(atoms, l_list, cutoff_radius, box)

    # Output to console
    if legacy:
        print("\nQ values for each l:")
        print("====================================")
        for ell in l_list:
            q_l = results[ell]['Q_l']
            print(f"Q_{ell} = {q_l:.4f}")
        print("====================================\n")
    else:
        print(f"\nCutoff radius: {cutoff_radius}")
        print("Q and global Q values for each l:")
        print("================================================")
        for ell in l_list:
            q_l = results[ell]['Q_l']
            q_l_global = results[ell].get('Q_l_global', None)
            if q_l_global is not None:
                print(f"l={ell}: Q_{ell} = {q_l:.4f}, Q_{ell}_global = {q_l_global:.4f}")
            else:
                print(f"l={ell}: Q_{ell} = {q_l:.4f}")
        print("================================================\n")
        # Check if the output folder exists, if not create it
        import os
        output_folder = "results"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        out_name = f"results/results_cutoff_{cutoff_radius}.txt"
        with open(out_name, 'w') as fout:
            fout.write(f"Cutoff radius: {cutoff_radius}\n")
            fout.write("l  Q_l  Q_l_global\n")
            for ell in l_list:
                q_l = results[ell]['Q_l']
                q_l_global = results[ell].get('Q_l_global', float('nan'))
                fout.write(f"{ell}  {q_l:.6f}  {q_l_global:.6f}\n")
        print(f"Results written to {out_name}\n")


def benchmark():
    """
    Benchmark the computation of Q_l for a large number of atoms.
    """
    try: 
        bounds , atoms = read_lammps_dump("data/benchmark_sc.dump")
    except FileNotFoundError:
        print("Benchmark file not found. Please run makeBenchMarkLattice.py first.")
        return
    
    n_cells = 30
    lattice_constant = 1.0
    r_cutoff = 1.2
    l_list = [4, 6, 8, 10, 12]
    atoms = compute_actual_positions(bounds, atoms)
    box_size = lattice_constant * n_cells
    box_bounds = torch.tensor([[0, box_size], [0, box_size], [0, box_size]], dtype=torch.float32)
    print(f"First 5 atom positions:")
    for i, atom in enumerate(atoms[:5]):
        print(f"Atom {i}: {atom['x']:.2f} {atom['y']:.2f} {atom['z']:.2f}")
    
    # Verify periodicity
    box_size = bounds[0][1] - bounds[0][0]
    print(f"\nBox size: {box_size:.2f}")
    print(f"Expected atoms: {n_cells**3}, Actual: {len(atoms)}")
    results = compute_Ql_from_atoms(atoms, l_list, r_cutoff, box_bounds)
    print("\nBenchmark Results:")
    print("====================================")
    for ell in l_list:
        q_l = results[ell]['Q_l']
        print(f"Q_{ell} = {q_l:.4f}, Q_l_global = {results[ell]['Q_l_global']:.4f}")
    print("====================================\n")
    # After computing results, for SC lattice:
    # https://hal.science/hal-03233572/document#:~:text=q4%200,696
    expected = {4: 0.764, 6: 0.354, 8: 0.718, 10: 0.411, 12: 0.696}
    for ell, Q_calc in results.items():
        assert torch.isclose(
            torch.tensor(Q_calc['Q_l']), 
            torch.tensor(expected[ell]), 
            atol=0.001
        ), f"Q_{ell} mismatch: {Q_calc['Q_l']} vs {expected[ell]}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute local and global bond order parameters Q_l for atoms from LAMMPS dump"
    )
    parser.add_argument(
        "-r", "--cutoff",
        type=float,
        help="filter atoms by radius before computing Q (if omitted, defaults to legacy r=0.5)"
    )
    args = parser.parse_args()
    benchmark()
    # main(args.cutoff)