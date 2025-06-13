import numpy as np

def generate_simple_cubic_lattice(n_cells_per_axis, lattice_constant):
    """
    Generate a simple cubic lattice with a given number of unit cells per axis
    and a specific lattice constant.
    """
    coords = []
    atom_id = 1
    atom_type = 1
    radius = 0.01 # Dummy value

    for x in range(n_cells_per_axis):
        for y in range(n_cells_per_axis):
            for z in range(n_cells_per_axis):
                coords.append([atom_id, atom_type,
                               x * lattice_constant,
                               y * lattice_constant,
                               z * lattice_constant,
                               radius])
                atom_id += 1

    return np.array(coords)

def write_dump_file(filename, atoms, box_bounds):
    """
    Write the atoms to a LAMMPS-style .dump file
    """
    with open(filename, "w") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write("104640000\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{len(atoms)}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for dim in box_bounds:
            f.write(f"{dim[0]} {dim[1]}\n")
        f.write("ITEM: ATOMS id type xs ys zs Radius\n")

        # Normalize coordinates to [0, 1)
        box_lengths = box_bounds[:, 1] - box_bounds[:, 0]
        for atom in atoms:
            atom_id, atom_type, x, y, z, radius = atom
            xs = (x - box_bounds[0][0]) / box_lengths[0]
            ys = (y - box_bounds[1][0]) / box_lengths[1]
            zs = (z - box_bounds[2][0]) / box_lengths[2]
            f.write(f"{int(atom_id)} {int(atom_type)} {xs:.16f} {ys:.16f} {zs:.16f} {radius:.1f}\n")

def sc():
    lattice_constant = 1.0
    n_cells = 30  # Explicitly set to get 27,000 atoms (30×30×30)
    atoms = generate_simple_cubic_lattice(n_cells, lattice_constant)
    
    box_size = lattice_constant * n_cells
    box_bounds = np.array([[0, box_size], [0, box_size], [0, box_size]])
    
    write_dump_file("benchmark_sc.dump", atoms, box_bounds)

def generate_fcc_lattice(n_cells_per_axis, lattice_constant):
    """
    Generate a face-centered cubic (FCC) lattice with a given number of
    unit cells per axis and a specific lattice constant.

    FCC basis (fractional):
        (0, 0, 0)
        (½, ½, 0)
        (½, 0, ½)
        (0, ½, ½)
    """
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ])

    coords = []
    atom_id = 1
    atom_type = 1
    radius = 0.5  # Dummy value

    for x in range(n_cells_per_axis):
        for y in range(n_cells_per_axis):
            for z in range(n_cells_per_axis):
                # Cartesian origin of this unit cell
                origin = np.array([x, y, z], dtype=float) * lattice_constant
                for frac in basis:
                    pos = origin + frac * lattice_constant
                    coords.append([
                        atom_id,
                        atom_type,
                        pos[0],
                        pos[1],
                        pos[2],
                        radius,
                    ])
                    atom_id += 1

    return np.array(coords)

def fcc():
    # Setup
    lattice_constant = 3.0
    n_cells = int(round((1500) ** (1 / 3)))  # ≈24
    atoms = generate_fcc_lattice(n_cells, lattice_constant)

    # Define box bounds
    box_size = lattice_constant * n_cells
    box_bounds = np.array(
        [
            [0, box_size],
            [0, box_size],
            [0, box_size],
        ]
    )

    write_dump_file("benchmark_fcc.dump", atoms, box_bounds)

if __name__ == "__main__":
    sc()
