import math
import numpy as np
from typing import List
from scipy.spatial import KDTree

D = 1.0  # Hard-sphere diameter (σ)

def generate_random_rotation() -> np.ndarray:
    """
    Generate a uniformly distributed random 3D rotation matrix.
    
    This function creates a random rotation in 3D space by combining three Euler angle
    rotations (two around the z-axis and one around the x-axis) with angles sampled
    uniformly between 0 and 2π.
    
    Returns:
        np.ndarray: A 3x3 rotation matrix representing a uniformly random 3D rotation.
    """
    theta, phi, psi = np.random.uniform(0, 2 * np.pi, size=3)
    rz1 = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                    [math.sin(theta),  math.cos(theta), 0.0],
                    [0.0,             0.0,             1.0]])
    rx  = np.array([[1.0, 0.0,            0.0           ],
                    [0.0, math.cos(phi), -math.sin(phi)],
                    [0.0, math.sin(phi),  math.cos(phi)]])
    rz2 = np.array([[math.cos(psi), -math.sin(psi), 0.0],
                    [math.sin(psi),  math.cos(psi), 0.0],
                    [0.0,           0.0,            1.0]])
    return rz2 @ rx @ rz1


def generate_perfect_crystal(crystal_type: str, num_particles: int) -> np.ndarray:
    """
    Generate positions of particles in a perfect crystal cluster centered at origin.
    
    Creates a perfect crystal lattice of the specified type with approximately the
    requested number of particles. The cluster is centered at the origin and then
    randomly rotated in 3D space.
    
    Args:
        crystal_type (str): Type of crystal lattice to generate. Must be one of:
                           "SC" (simple cubic), "BCC" (body-centered cubic),
                           or "FCC" (face-centered cubic).
        num_particles (int): Target number of particles in the cluster. The actual
                            number may be slightly larger to complete unit cells.
    
    Returns:
        np.ndarray: An Nx3 array of particle positions for the generated cluster,
                    where N is approximately num_particles.
    
    Raises:
        ValueError: If an unknown crystal type is specified.
    """
    positions: List[np.ndarray] = []

    if crystal_type == "SC":
        n = math.ceil(num_particles ** (1 / 3))
        a = D  
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if len(positions) < num_particles:
                        positions.append(np.array([i * a, j * a, k * a]))

    elif crystal_type == "BCC":
        a = 2 * D / math.sqrt(3)
        n = math.ceil((num_particles / 2) ** (1 / 3))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if len(positions) < num_particles:
                        cell = np.array([i * a, j * a, k * a])
                        positions.append(cell)
                        if len(positions) < num_particles:
                            positions.append(cell + np.array([a / 2] * 3))

    elif crystal_type == "FCC":
        a = D * math.sqrt(2)
        n = math.ceil((num_particles / 4) ** (1 / 3))
        offsets = np.array([[0, 0, 0],
                            [0.5, 0.5, 0],
                            [0.5, 0, 0.5],
                            [0, 0.5, 0.5]])
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for off in offsets:
                        if len(positions) < num_particles:
                            positions.append(np.array([i, j, k]) * a + off * a)
    else:
        raise ValueError(f"Unknown crystal type: {crystal_type}")

    positions = np.asarray(positions)

    positions -= positions.mean(axis=0)
    positions = positions @ generate_random_rotation().T
    return positions


def try_place_cluster(crystal_type: str, size: int, box_size: float,
                      existing_positions: np.ndarray, max_attempts: int = 200) -> np.ndarray:
    """
    Attempt to place a crystal cluster in the simulation box without overlaps.
    
    Generates a crystal cluster and attempts to place it in the simulation box such
    that it doesn't overlap with any existing particles. The cluster is placed at a
    random position that ensures it fits entirely within the box and is given periodic
    boundary conditions.
    
    Args:
        crystal_type (str): Type of crystal lattice ("SC", "BCC", or "FCC").
        size (int): Number of particles in the cluster.
        box_size (float): Size of the cubic simulation box.
        existing_positions (np.ndarray): Nx3 array of existing particle positions.
        max_attempts (int, optional): Maximum number of placement attempts before
                                      giving up. Defaults to 200.
    
    Returns:
        np.ndarray: Positions of the successfully placed cluster particles.
    
    Raises:
        RuntimeError: If the cluster cannot be placed without overlaps after
                     max_attempts tries. This suggests the system is too dense
                     or the cluster is too large.
    """
    rng = np.random.default_rng()
    for _ in range(max_attempts):
        cluster = generate_perfect_crystal(crystal_type, size)
        r_max = np.linalg.norm(cluster, axis=1).max()

        centre = rng.uniform(r_max, box_size - r_max, size=3)
        cluster += centre
        cluster = np.mod(cluster, box_size)

        if existing_positions.size == 0:
            return cluster        
        # Ensure existing positions are within the periodic box
        existing_positions_wrapped = np.mod(existing_positions, box_size)
        tree = KDTree(existing_positions_wrapped, boxsize=box_size)
        dists, _ = tree.query(cluster, k=1)
        if np.all(dists >= D):
            return cluster

    raise RuntimeError("Could not place cluster without overlap - try reducing density or cluster size")

def plot_crystal(ax, positions, title):
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', marker='o')
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Test random rotation generation
    rotations = [generate_random_rotation() for _ in range(3)]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Random Rotations")
    ax.set_xlabel("X") 
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    for rot in rotations:
        # Create a unit cube to visualize the rotation
        cube = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
        rotated_cube = cube @ rot.T
        ax.scatter(rotated_cube[:, 0], rotated_cube[:, 1], rotated_cube[:, 2])
        ax.plot_trisurf(rotated_cube[:, 0], rotated_cube[:, 1], rotated_cube[:, 2], alpha=0.1)
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    # Test perfect crystal generation
    sc = generate_perfect_crystal("SC", 27)
    bcc = generate_perfect_crystal("BCC", 32)
    fcc = generate_perfect_crystal("FCC", 64)
    
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    plot_crystal(ax1, sc, "Simple Cubic (SC)")
    ax2 = fig.add_subplot(132, projection='3d')
    plot_crystal(ax2, bcc, "Body-Centred Cubic (BCC)")
    ax3 = fig.add_subplot(133, projection='3d')
    plot_crystal(ax3, fcc, "Face-Centred Cubic (FCC)")
    plt.show()

    # Test cluster placement when there is another crystal in the box
    box_size = 30.0
    # Place existing crystals in the box properly
    existing_positions = np.vstack((sc + 5.0, bcc + 5.0))  # Center them in the box
    existing_positions = np.mod(existing_positions, box_size)  # Ensure they're within bounds
    
    try:
        new_cluster = try_place_cluster("FCC", 64, box_size, existing_positions)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_crystal(ax, new_cluster, "Placed FCC Cluster")
        ax.scatter(existing_positions[:, 0], existing_positions[:, 1], existing_positions[:, 2], c='red', marker='x', label='Existing Crystals')
        ax.legend()
        plt.show()
    except RuntimeError as e:
        print(f"Error placing cluster: {e}")
        
    # Test cluster placement when there are no existing positions
    try:
        new_cluster = try_place_cluster("FCC", 64, box_size, np.empty((0, 3)))
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_crystal(ax, new_cluster, "Placed FCC Cluster (No Existing Positions)")
        plt.show()
    except RuntimeError as e:
        print(f"Error placing cluster: {e}")