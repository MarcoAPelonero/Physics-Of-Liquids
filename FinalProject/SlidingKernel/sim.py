from utils.importUtils import read_lammps_dump, compute_actual_positions
from utils.parallelQ import compute_Ql_from_atoms
from utils.kernelSep import split_atoms_by_kernel
from tqdm import tqdm
import json
import argparse
import numpy as np
import torch

def main(filename, cutoff_radius=1.4, l_values=None, verbose=False):
    if l_values is None:
        l_values = [2, 4, 6, 8, 10]

    # Read global box and atoms
    box_bounds, atoms = read_lammps_dump(filename)
    atoms = compute_actual_positions(box_bounds, atoms)

    # Convert global bounds to numpy for convenience (not actually used below)
    box_bounds = np.array(box_bounds)

    # Kernel parameters
    kernel_fraction = 0.2   # 20% of box side
    overlap_fraction = 0.5  # 50% overlap

    # Compute kernel “size” in each dimension
    Lx = box_bounds[0,1] - box_bounds[0,0]
    Ly = box_bounds[1,1] - box_bounds[1,0]
    Lz = box_bounds[2,1] - box_bounds[2,0]
    k_size = (kernel_fraction * Lx,
              kernel_fraction * Ly,
              kernel_fraction * Lz)

    # Stride with 50% overlap
    stride = (k_size[0] * (1 - overlap_fraction),
              k_size[1] * (1 - overlap_fraction),
              k_size[2] * (1 - overlap_fraction))

    print("l-values:", l_values)
    all_kernels = split_atoms_by_kernel(atoms,
                                        k_size,
                                        stride=stride,
                                        bounds=box_bounds)
    
    all_kernels = list(split_atoms_by_kernel(
        atoms,
        k_size,
        stride=stride,
        bounds=box_bounds
    ))

    for kernel in tqdm(all_kernels, total=len(all_kernels),
                       desc="Processing Kernels",
                       position=1, leave=True):
        # kernel["box_size"] is just the lengths (Lx, Ly, Lz) of this sub-box
        sub_Lx, sub_Ly, sub_Lz = kernel["box_size"]
        cx, cy, cz = kernel["center"]

        # Reconstruct the min/max bounds of the sub-box
        sub_bounds = np.array([
            [cx - sub_Lx/2, cx + sub_Lx/2],
            [cy - sub_Ly/2, cy + sub_Ly/2],
            [cz - sub_Lz/2, cz + sub_Lz/2]
        ], dtype=float)

        # **Here’s the key change:** convert to torch.Tensor
        sub_bounds = torch.tensor(sub_bounds, dtype=torch.float32)
        # Now pass that 3×2 array into compute_Ql_from_atoms
        results = compute_Ql_from_atoms(
            kernel["atoms"],
            l_list=l_values,
            r_cutoff=cutoff_radius,
            box=sub_bounds
        )

        # Print & store
        for ell in l_values:
            q_l       = results[ell]['Q_l']
            q_l_glob  = results[ell].get('Q_l_global')
            if verbose:
                if q_l_glob is not None:
                    print(f"Center {kernel['center']}, l={ell}: Q_{ell} = {q_l:.4f}, Q_{ell}_global = {q_l_glob:.4f}")
                else:
                    print(f"Center {kernel['center']}, l={ell}: Q_{ell} = {q_l:.4f}")
            kernel.setdefault("results", {})[ell] = {
                'Q_l': q_l,
                'Q_l_global': q_l_glob
            }

    # Dump everything (you might want to collect all_kernels rather than atoms here,
    # depending on what structure you actually want to save)
    outname = (
        f"results_cutoff{cutoff_radius}"
        f"_l{'-'.join(map(str, l_values))}"
        f"_kernel{kernel_fraction}"
        f"_stride{overlap_fraction}.json"
    )
    with open(outname, 'w') as f:
        json.dump(all_kernels, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run sliding‐kernel Ql analysis on a LAMMPS dump"
    )
    parser.add_argument("dump_file", nargs='?', default="last.dump",
                        help="Dump file to read from")
    parser.add_argument("-c", "--cutoff_radius", type=float, default=1.4,
                        help="Cutoff radius (default: 1.4)")
    parser.add_argument("-l", "--l_values", default="4,6,8,12",
                        help="Comma-separated list of l-values (default: 4,6,8,12)")
    args = parser.parse_args()

    l_vals = list(map(int, args.l_values.split(",")))
    main(args.dump_file, cutoff_radius=args.cutoff_radius, l_values=l_vals)
