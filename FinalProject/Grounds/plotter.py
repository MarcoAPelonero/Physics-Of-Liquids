import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from utils import read_lammps_dump

def scale_positions(atoms, box_bounds):
    """
    Converts scaled positions (0-1) to absolute coordinates using box bounds.
    """
    xlo, xhi = box_bounds[0]
    ylo, yhi = box_bounds[1]
    zlo, zhi = box_bounds[2]
    for a in atoms:
        a['x'] = xlo + a['x_scaled'] * (xhi - xlo)
        a['y'] = ylo + a['y_scaled'] * (yhi - ylo)
        a['z'] = zlo + a['z_scaled'] * (zhi - zlo)

def plot_spheres(atoms, box_bounds, sphere_resolution=12, save__3d_file=False):
    """
    Plots each atom as a sphere in 3D. 
    sphere_resolution controls the mesh density of each sphere.
    """
    types = sorted({a['type'] for a in atoms})
    cmap = plt.get_cmap('tab10')
    colors = {t: cmap(i % cmap.N) for i, t in enumerate(types)}

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1,1,1))

    u = np.linspace(0, 2 * np.pi, sphere_resolution, endpoint=False)
    v = np.linspace(0, np.pi, sphere_resolution)
    uu, vv = np.meshgrid(u, v)
    xs_unit = np.cos(uu) * np.sin(vv)
    ys_unit = np.sin(uu) * np.sin(vv)
    zs_unit = np.cos(vv)

    for atom in tqdm(atoms, desc="Plotting spheres"):
        r = atom['radius']
        x0, y0, z0 = atom['x'], atom['y'], atom['z']
        x_s = x0 + r * xs_unit
        y_s = y0 + r * ys_unit
        z_s = z0 + r * zs_unit
        ax.plot_surface(x_s, y_s, z_s, rstride=1, cstride=1, 
                        color=colors[atom['type']], linewidth=0, shade=True, alpha=0.9)

    ax.set_xlim(box_bounds[0])
    ax.set_ylim(box_bounds[1])
    ax.set_zlim(box_bounds[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()

    if save__3d_file:
        save_obj_spheres(atoms, sphere_resolution, filename="spheres.obj")
    
    plt.show()

def save_obj_spheres(atoms, sphere_resolution, filename="spheres.obj"):
    """
    Exports sphere geometries for each atom into an OBJ file.
    Each sphere is created as a grid mesh based on the sphere_resolution.
    """
    u = np.linspace(0, 2 * np.pi, sphere_resolution, endpoint=False)
    v = np.linspace(0, np.pi, sphere_resolution)
    uu, vv = np.meshgrid(u, v)
    xs_unit = np.cos(uu) * np.sin(vv)
    ys_unit = np.sin(uu) * np.sin(vv)
    zs_unit = np.cos(vv)
    n_u = sphere_resolution
    n_v = sphere_resolution

    vertex_offset = 1  
    with open(filename, "w") as f:
        f.write("# OBJ file containing sphere geometries for atoms\n")
        for atom in atoms:
            r = atom['radius']
            x0, y0, z0 = atom['x'], atom['y'], atom['z']
            for i in range(n_v):
                for j in range(n_u):
                    x = x0 + r * xs_unit[i, j]
                    y = y0 + r * ys_unit[i, j]
                    z = z0 + r * zs_unit[i, j]
                    f.write(f"v {x} {y} {z}\n")
            for i in range(n_v - 1):
                for j in range(n_u):
                    next_j = (j + 1) % n_u 
                    v1 = vertex_offset + i * n_u + j
                    v2 = vertex_offset + (i + 1) * n_u + j
                    v3 = vertex_offset + (i + 1) * n_u + next_j
                    v4 = vertex_offset + i * n_u + next_j
                    f.write(f"f {v1} {v2} {v3}\n")
                    f.write(f"f {v1} {v3} {v4}\n")
            vertex_offset += n_u * n_v  

if __name__ == "__main__":
    dump_file = "benchmark_sc.dump"           
    sphere_mesh_resolution = 16       

    box_bounds, atoms = read_lammps_dump(dump_file)
    scale_positions(atoms, box_bounds)
    plot_spheres(atoms, box_bounds, sphere_resolution=sphere_mesh_resolution)