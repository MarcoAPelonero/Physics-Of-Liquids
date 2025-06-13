from utils import read_lammps_dump
from utils import compute_extents, print_average_scaled_coordinates

bounds, atoms = read_lammps_dump("last.dump")
x_range, y_range, z_range = compute_extents(bounds, atoms)
print("x:", x_range, "y:", y_range, "z:", z_range)
print_average_scaled_coordinates(atoms)