from typing import List
import math
import os
import csv
import numpy as np
from scipy.spatial import KDTree
import torch

D = 1.0  # Hard-sphere diameter (σ)

def generate_disordered_particles(num_particles: int, fixed_positions: np.ndarray,
                                  box_size: float) -> np.ndarray:
    """Grid‑initialise fluid particles, avoiding overlap with *fixed_positions*."""
    if num_particles == 0:
        return np.empty((0, 3))

    grid_spacing = max(D, box_size / math.ceil(num_particles ** (1 / 3)))
    grid_points = int(box_size / grid_spacing)

    xs = np.linspace(0, box_size, grid_points, endpoint=False)
    grid = np.vstack(np.meshgrid(xs, xs, xs)).reshape(3, -1).T

    if fixed_positions.size:
        tree = KDTree(fixed_positions, boxsize=box_size)
        dists, _ = tree.query(grid, k=1)
        grid = grid[dists >= D]

    np.random.shuffle(grid)
    return grid[:num_particles]

def mc_displace(positions: np.ndarray, mobile_indices: List[int], box_size: float,
                n_steps_per_particle: int, step_size: float) -> np.ndarray:
    """General MC displacement for specified particles."""

    positions = np.mod(positions, box_size)
    tree = KDTree(positions, boxsize=box_size)
    adaptive = step_size
    accepted = 0
    total_steps = n_steps_per_particle * len(mobile_indices)
    
    for step in range(total_steps):
        idx = np.random.choice(mobile_indices)
        old = positions[idx].copy()
        trial = old + (np.random.rand(3) - 0.5) * adaptive
        trial = np.mod(trial, box_size)

        if step % 100 == 0:
            tree = KDTree(positions, boxsize=box_size)
        
        neighbours = tree.query_ball_point(trial, r=1.1 * D) 
        neighbours = [j for j in neighbours if j != idx]
        valid = True
        for j in neighbours:
            delta = trial - positions[j]
            delta -= box_size * np.round(delta / box_size)
            if np.linalg.norm(delta) < D:
                valid = False
                break

        if step > 0 and step % 1000 == 0:
            rate = accepted / 1000
            if rate < 0.20:
                adaptive *= 0.9
            elif rate > 0.40:
                adaptive = min(adaptive * 1.1, D)
            accepted = 0

        if valid:
            positions[idx] = trial
            accepted += 1

    return positions

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Generate disordered particles
    num_particles = 100
    box_size = 20.0
    fixed_positions = np.array([[1, 1, 1], [2, 2, 2]])  # Example fixed positions
    disordered_particles = generate_disordered_particles(num_particles, fixed_positions, box_size)
    
    # Print actual number of particles generated
    actual_num_particles = len(disordered_particles)
    print(f"Requested: {num_particles}, Generated: {actual_num_particles}")
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(disordered_particles[:, 0], disordered_particles[:, 1], disordered_particles[:, 2], c='blue', marker='o')
    ax.set_title("Disordered Particles")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    plt.show()

    # On the same configuration, test the MC displacement
    mobile_indices = list(range(actual_num_particles))  
    n_steps_per_particle = 250
    step_size = 0.15 * D
    displaced_particles = mc_displace(disordered_particles.copy(), mobile_indices, box_size, n_steps_per_particle, step_size)
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(disordered_particles[:, 0], disordered_particles[:, 1], disordered_particles[:, 2], c='blue', marker='o', label='Before MC')
    ax.scatter(displaced_particles[:, 0], displaced_particles[:, 1], displaced_particles[:, 2], c='red', marker='x', label='After MC')
    ax.set_title("MC Displacement of Particles")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.show()