#!/usr/bin/env python3
import os
import argparse
import torch

from utils.importUtils import (
    read_lammps_dump,
    compute_actual_positions,
    filter_atoms_by_radius,
)
from utils.parallelQ import compute_Ql_knn_from_atoms  # <-- NEW import


# ──────────────────────────────────────────────────────────────────────────────
def main(n_neighbors: int = 6):
    """
    Compute local and global Q_l using a fixed number of nearest neighbours.
    """
    # ── load snapshot ─────────────────────────────────────────────────────────
    bounds, atoms = read_lammps_dump("data/last.dump")
    atoms = compute_actual_positions(bounds, atoms)

    # ── optional preprocessing (unchanged) ───────────────────────────────────
    atoms = filter_atoms_by_radius(atoms, 0.5)

    # ── parameters ───────────────────────────────────────────────────────────
    l_list = [4, 6, 8, 10, 12]
    box    = torch.tensor(bounds, dtype=torch.float32)

    # ── compute order parameters ─────────────────────────────────────────────
    results = compute_Ql_knn_from_atoms(atoms, l_list, n_neighbors, box)

    # ── console output ───────────────────────────────────────────────────────
    print(f"\nNearest neighbours per atom: {n_neighbors}")
    print("Q and global-Q values for each l")
    print("================================================")
    for ell in l_list:
        q_l         = results[ell]["Q_l"]
        q_l_global  = results[ell]["Q_l_global"]
        print(f"l={ell}: Q_{ell} = {q_l:.4f}, Q_{ell}_global = {q_l_global:.4f}")
    print("================================================\n")

    # ── save to file ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_name = f"results/results_knn_{n_neighbors}.txt"
    with open(out_name, "w") as fout:
        fout.write(f"Nearest neighbours: {n_neighbors}\n")
        fout.write("l  Q_l  Q_l_global\n")
        for ell in l_list:
            fout.write(
                f"{ell}  {results[ell]['Q_l']:.6f}  {results[ell]['Q_l_global']:.6f}\n"
            )
    print(f"Results written to {out_name}\n")


# ──────────────────────────────────────────────────────────────────────────────
def benchmark():
    """
    Quick sanity check on a simple-cubic lattice using 4 neighbours.
    """
    try:
        bounds, atoms = read_lammps_dump("data/benchmark_sc.dump")
    except FileNotFoundError:
        print("Benchmark file not found. Please run makeBenchMarkLattice.py first.")
        return

    # ── lattice / algorithm settings ─────────────────────────────────────────
    n_cells      = 30
    lattice_const = 1.0
    n_neighbors  = 6
    l_list       = [4, 6, 8, 10, 12]

    atoms = compute_actual_positions(bounds, atoms)
    box_size   = lattice_const * n_cells
    box_bounds = torch.tensor(
        [[0, box_size], [0, box_size], [0, box_size]], dtype=torch.float32
    )

    # ── basic sanity prints ──────────────────────────────────────────────────
    print("First 5 atom positions:")
    for i, atom in enumerate(atoms[:5]):
        print(f"Atom {i}: {atom['x']:.2f} {atom['y']:.2f} {atom['z']:.2f}")

    print(f"\nBox size: {box_size:.2f}")
    print(f"Expected atoms: {n_cells**3}, Actual: {len(atoms)}")

    # ── compute order parameters ─────────────────────────────────────────────
    results = compute_Ql_knn_from_atoms(atoms, l_list, n_neighbors, box_bounds)

    print("\nBenchmark Results:")
    print("====================================")
    for ell in l_list:
        print(
            f"Q_{ell} = {results[ell]['Q_l']:.4f}, "
            f"Q_{ell}_global = {results[ell]['Q_l_global']:.4f}"
        )
    print("====================================\n")

    # ── reference check (values from literature, still valid for k=4) ───────
    expected = {4: 0.764, 6: 0.354, 8: 0.718, 10: 0.411, 12: 0.696}
    for ell in l_list:
        Q_calc = results[ell]["Q_l"]
        assert torch.isclose(
            torch.tensor(Q_calc),
            torch.tensor(expected[ell]),
            atol=0.001,
        ), f"Q_{ell} mismatch: {Q_calc} vs {expected[ell]}"


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute local and global bond-order parameters Q_l using a fixed "
            "number of nearest neighbours per atom."
        )
    )
    parser.add_argument(
        "-k",
        "--knn",
        type=int,
        default=6,
        help="number of nearest neighbours to use (default: 6)",
    )
    args = parser.parse_args()

    main(args.knn)
    # Uncomment to run the benchmark
    # If you want the benchmark to run automatically, uncomment:
    # benchmark()
