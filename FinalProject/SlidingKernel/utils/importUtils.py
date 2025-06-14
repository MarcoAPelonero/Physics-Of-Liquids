def read_lammps_dump(filename):
    """
    Reads a LAMMPS dump file and returns:
      - box_bounds: [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
      - atoms: list of dicts with keys ['id','type','x','y','z','radius']
    Assumes the 'xs ys zs' columns are scaled (0–1) and box is orthogonal.
    """
    atoms = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith("ITEM: BOX BOUNDS"):
                bounds = [list(map(float, f.readline().split())) for _ in range(3)]
            elif line.startswith("ITEM: ATOMS"):
                cols = line.strip().split()[2:] 
                for atom_line in f:
                    vals = atom_line.split()
                    atom = dict(zip(cols, map(float, vals)))
                    atoms.append({
                        'id': int(atom['id']),
                        'type': int(atom['type']),
                        'x_scaled': atom['xs'],
                        'y_scaled': atom['ys'],
                        'z_scaled': atom['zs'],
                        'radius': atom['Radius']
                    })
                break
            line = f.readline()
    return bounds, atoms

def compute_actual_positions(bounds, atoms):
    """
    Given box_bounds and atoms from read_lammps_dump, computes the actual x, y, and z positions
    from the scaled coordinates and returns a new list of atom dictionaries with the non-scaled
    positions added as 'x', 'y', and 'z'.
    """
    x_bounds, y_bounds, z_bounds = bounds
    xlo, xhi = x_bounds
    ylo, yhi = y_bounds
    zlo, zhi = z_bounds

    new_atoms = []
    for atom in atoms:
        actual_x = atom['x_scaled'] * (xhi - xlo) + xlo
        actual_y = atom['y_scaled'] * (yhi - ylo) + ylo
        actual_z = atom['z_scaled'] * (zhi - zlo) + zlo
        new_atom = atom.copy()
        new_atom['x'] = actual_x
        new_atom['y'] = actual_y
        new_atom['z'] = actual_z
        new_atoms.append(new_atom)
    return new_atoms

def compute_extents(bounds, atoms):
    """
    Given box_bounds and atoms from read_lammps_dump, computes the actual x, y, z positions
    from the scaled coordinates and returns the min and max for each direction.
    
    Returns a tuple of ((min_x, max_x), (min_y, max_y), (min_z, max_z)).
    """
    x_bounds, y_bounds, z_bounds = bounds
    xlo, xhi = x_bounds
    ylo, yhi = y_bounds
    zlo, zhi = z_bounds

    x_positions = [atom['x_scaled'] * (xhi - xlo) + xlo for atom in atoms]
    y_positions = [atom['y_scaled'] * (yhi - ylo) + ylo for atom in atoms]
    z_positions = [atom['z_scaled'] * (zhi - zlo) + zlo for atom in atoms]

    min_x, max_x = min(x_positions), max(x_positions)
    min_y, max_y = min(y_positions), max(y_positions)
    min_z, max_z = min(z_positions), max(z_positions)
    
    return (min_x, max_x), (min_y, max_y), (min_z, max_z)

def print_average_scaled_coordinates(atoms):
    """
    Computes and prints the average of the scaled x, y, and z coordinates.
    """
    if not atoms:
        print("No atoms found.")
        return
    avg_x = sum(atom['x_scaled'] for atom in atoms) / len(atoms)
    avg_y = sum(atom['y_scaled'] for atom in atoms) / len(atoms)
    avg_z = sum(atom['z_scaled'] for atom in atoms) / len(atoms)
    print("Average scaled coordinates - x: {:.4f}, y: {:.4f}, z: {:.4f}".format(avg_x, avg_y, avg_z))

def filter_atoms_by_radius(atoms, target_radius=0.5):
    """
    Filters the atoms list to only include atoms with the specified target_radius.
    
    Parameters:
      atoms (list): List of atom dictionaries.
      target_radius (float): The radius value to filter by (default is 0.5).
      
    Returns:
      list: A list of atoms with radius equal to target_radius.
    """
    return [atom for atom in atoms if abs(atom['radius'] - target_radius) < 1e-6]
 