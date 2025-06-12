from utils.importUtils import read_lammps_dump, compute_actual_positions
import sys

def filter_atoms_in_subvolume(bounds, atoms, frac):
    """
    Filter atoms inside a cube of fractional volume `frac`, centered at the center of the box.
    """
    x_bounds, y_bounds, z_bounds = bounds
    xlo, xhi = x_bounds
    ylo, yhi = y_bounds
    zlo, zhi = z_bounds

    x_center = (xlo + xhi) / 2
    y_center = (ylo + yhi) / 2
    z_center = (zlo + zhi) / 2

    x_len = xhi - xlo
    y_len = yhi - ylo
    z_len = zhi - zlo

    dx = frac * x_len / 2
    dy = frac * y_len / 2
    dz = frac * z_len / 2

    new_atoms = []
    for atom in compute_actual_positions(bounds, atoms):
        if (x_center - dx <= atom['x'] <= x_center + dx and
            y_center - dy <= atom['y'] <= y_center + dy and
            z_center - dz <= atom['z'] <= z_center + dz):
            new_atoms.append(atom)
    return new_atoms

def write_lammps_dump(filename, bounds, atoms, frac):
    with open(filename, 'w') as f:
        f.write("ITEM: TIMESTEP\n")
        f.write("104640000\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{len(atoms)}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for lo, hi in bounds:
            f.write(f"{lo * frac} {hi * frac}\n")
        f.write("ITEM: ATOMS id type xs ys zs Radius\n")
        for atom in atoms:
            f.write(f"{atom['id']} {atom['type']} {atom['x_scaled']} {atom['y_scaled']} {atom['z_scaled']} {atom['radius']}\n")

if __name__ == "__main__":
    input_file = "last.dump"
    output_file = "filtered.dump"
    frac = 0.3  

    bounds, atoms = read_lammps_dump(input_file)
    filtered_atoms = filter_atoms_in_subvolume(bounds, atoms, frac)
    write_lammps_dump(output_file, bounds, filtered_atoms, frac)