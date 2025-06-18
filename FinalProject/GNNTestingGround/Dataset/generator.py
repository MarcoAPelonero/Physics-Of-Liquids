# hard_sphere_mini_clusters.py
"""
Generate an ensemble of hard‐sphere configurations that contain several *mini* crystalline
clusters embedded in a dense fluid.  Each configuration can carry:

* 1 – `num_clusters_max` randomly placed clusters (SC/BCC/FCC)
* Cluster sizes drawn from [`cluster_min_particles`, `cluster_max_particles`]
* A configurable fraction of the cluster particles re‑labelled as fluid so they can
  diffuse into (and out of) the crystal during MC, giving noisy interfaces.

All user‑visible parameters are collected at the top of the script so you can tweak the
behaviour without modifying the rest of the code.
"""

from __future__ import annotations

import math
import os
import random
import csv
import multiprocessing
from functools import partial
from typing import List

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

# -----------------------------------------------------------------------------
#  GLOBAL PARAMETERS  ──────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------
# Sphere and system sizes 
D: float = 1.0          # Hard‑sphere diameter (σ)
ETA: float = 0.45       # Packing fraction ϕ
N_PARTICLES: int = 1_000 # Total particles per configuration

# Crystalline mini‑clusters
NUM_CLUSTERS_MIN: int = 1   # At least one cluster in an *ordered* sample
NUM_CLUSTERS_MAX: int = 4   # Increase/decrease to get sparser or denser seeds
CLUSTER_MIN_PARTICLES: int = 25
CLUSTER_MAX_PARTICLES: int = 60

#  Monte‑Carlo parameters
MC_STEPS_PER_PARTICLE: int = 250   # How long we randomise the *fluid* particles
STEP_SIZE: float = 0.15 * D        # Initial displacement amplitude (adaptive)

#  Noise at the crystal–fluid interface
INSIDE_NOISE_FRACTION: float = 0.10  # Fraction of cluster sites relabelled as fluid

#  Output: how many configs to generate and where to put them
NUM_SAMPLES: int = 1000
OUTPUT_DIR: str = "hard_sphere_ensemble"
RNG_BASE_SEED: int = 42

# -----------------------------------------------------------------------------
#  HELPER FUNCTIONS  ───────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_random_rotation() -> np.ndarray:
    """Return a uniformly distributed 3‑D rotation matrix."""
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


# -----------------------------------------------------------------------------
#  CRYSTAL SEEDS  ──────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------


def generate_perfect_crystal(crystal_type: str, num_particles: int) -> np.ndarray:
    """Build a perfect SC/BCC/FCC cluster centred at the origin."""
    positions: List[np.ndarray] = []

    if crystal_type == "SC":
        # simple cubic – one lattice point / cell
        n = math.ceil(num_particles ** (1 / 3))
        a = D  # lattice constant so nearest neighbours sit at distance σ
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if len(positions) < num_particles:
                        positions.append(np.array([i * a, j * a, k * a]))

    elif crystal_type == "BCC":
        # body‑centred cubic – two points / cell
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
        # face‑centred cubic – four points / cell
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

    # Centre and randomly rotate
    positions -= positions.mean(axis=0)
    positions = positions @ generate_random_rotation().T
    return positions


# Try placing a cluster in the simulation box without overlapping previous ones

def try_place_cluster(crystal_type: str, size: int, box_size: float,
                      existing_positions: np.ndarray, max_attempts: int = 200) -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(max_attempts):
        cluster = generate_perfect_crystal(crystal_type, size)
        r_max = np.linalg.norm(cluster, axis=1).max()

        # Pick a random centre far enough from the box edges so the cluster fits
        centre = rng.uniform(r_max, box_size - r_max, size=3)
        cluster += centre
        cluster = np.mod(cluster, box_size)

        # Check hard‑sphere overlap with existing crystalline particles only;
        # fluid particles will be inserted later via grid initialisation
        if existing_positions.size == 0:
            return cluster
        tree = KDTree(existing_positions, boxsize=box_size)
        dists, _ = tree.query(cluster, k=1)
        if np.all(dists >= D):
            return cluster

    raise RuntimeError("Could not place cluster without overlap – try reducing density or cluster size")


# -----------------------------------------------------------------------------
#  DISORDERED FLUID  ───────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
#  MONTE‑CARLO  ────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------


def mc_displace_fluid(positions: np.ndarray, is_crystal: np.ndarray, box_size: float,
                      n_steps_per_particle: int, step_size: float) -> np.ndarray:
    """Randomise *fluid* (is_crystal == 0) particles with a simple Metropolis MC."""
    mobile = np.where(is_crystal == 0)[0]
    if mobile.size == 0:
        return positions

    positions = np.mod(positions, box_size)
    tree = KDTree(positions, boxsize=box_size)

    adaptive = step_size
    accepted: int = 0

    for step in range(n_steps_per_particle * mobile.size):
        idx = np.random.choice(mobile)
        old = positions[idx].copy()
        trial = old + (np.random.rand(3) - 0.5) * adaptive
        trial = np.mod(trial, box_size)

        neighbours = tree.query_ball_point(trial, r=2 * D)
        neighbours = [j for j in neighbours if j != idx]
        valid = True
        for j in neighbours:
            delta = trial - positions[j]
            delta -= box_size * np.round(delta / box_size)
            if np.linalg.norm(delta) < D:
                valid = False
                break

        # adapt every 1 000 attempted moves
        if step and step % 1_000 == 0:
            rate = accepted / 1_000
            if rate < 0.20:
                adaptive *= 0.9
            elif rate > 0.40:
                adaptive = min(adaptive * 1.1, D)
            accepted = 0

        if valid:
            positions[idx] = trial
            accepted += 1
            tree = KDTree(positions, boxsize=box_size)

    return positions


# -----------------------------------------------------------------------------
#  CONFIGURATION I/O  ─────────────────────────────────────────────────────────-
# -----------------------------------------------------------------------------


def save_configuration(index: int, positions: np.ndarray, is_crystal: np.ndarray,
                       box_size: float, cluster_info: str) -> None:
    fname = os.path.join(OUTPUT_DIR, f"config_{index:03d}.csv")
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# Box size (L): {box_size:.6f}"])
        w.writerow([f"# Clusters: {cluster_info}"])
        w.writerow(["x", "y", "z", "crystal_marker"])
        for pos, marker in zip(positions, is_crystal):
            w.writerow([*pos, int(marker)])


# -----------------------------------------------------------------------------
#  MAIN WORKER  ────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------


def generate_configuration(index: int, params: dict) -> None:
    max_retries = 10
    tries = 0
    while tries < max_retries:
        try:
            rng_seed = params["seed"] + index
            np.random.seed(rng_seed)
            random.seed(rng_seed)

            L: float = params["box_size"]

            # Half of the ensemble purely disordered, half with crystal seeds
            ordered_sample = index >= params["num_samples"] // 2

            crystal_positions: List[np.ndarray] = []
            cluster_descriptions: List[str] = []

            if ordered_sample:
                # determine how many clusters and make sure we never exceed particle budget
                remaining = N_PARTICLES
                n_clusters = random.randint(NUM_CLUSTERS_MIN, NUM_CLUSTERS_MAX)

                for _ in range(n_clusters):
                    if remaining < CLUSTER_MIN_PARTICLES:
                        break  # not enough room left for another crystal
                    c_type = random.choice(["SC", "BCC", "FCC"])
                    c_size = random.randint(CLUSTER_MIN_PARTICLES,
                                             min(CLUSTER_MAX_PARTICLES, remaining))
                    existing = np.vstack(crystal_positions) if crystal_positions else np.empty((0, 3))
                    placed = try_place_cluster(c_type, c_size, L, existing)
                    crystal_positions.append(placed)
                    cluster_descriptions.append(f"{c_type}:{c_size}")
                    remaining -= c_size

                crystal_positions = np.vstack(crystal_positions) if crystal_positions else np.empty((0, 3))
            else:
                # purely fluid sample
                remaining = N_PARTICLES
                crystal_positions = np.empty((0, 3))

            # Introduce *inside* noise by relabelling a fraction of cluster sites as fluid
            n_noise = int(INSIDE_NOISE_FRACTION * crystal_positions.shape[0])
            noise_indices = np.random.choice(crystal_positions.shape[0], n_noise, replace=False) if n_noise > 0 else []

            # Build full particle list: crystal + fluid
            n_fluid = remaining
            fluid_positions = generate_disordered_particles(n_fluid, crystal_positions, L)

            positions = np.vstack([crystal_positions, fluid_positions])
            is_crystal = np.concatenate([
                np.ones(crystal_positions.shape[0], dtype=int),
                np.zeros(fluid_positions.shape[0], dtype=int),
            ])
            if n_noise:
                is_crystal[noise_indices] = 0  # those crystal particles become mobile

            # MC to decorrelate the *fluid* particles (including noisy ones)
            positions = mc_displace_fluid(positions, is_crystal, L,
                                          MC_STEPS_PER_PARTICLE, STEP_SIZE)

            save_configuration(index, positions, is_crystal, L,
                               ", ".join(cluster_descriptions) if cluster_descriptions else "None")
            return

        except RuntimeError as e:
            tries += 1
            print(f"Config {index}: Simulation error encountered: {e}. Retrying (attempt {tries})...")

    # Fallback: if max_retries reached, generate a purely fluid configuration.
    print(f"Config {index}: Simulation failed after {max_retries} attempts, using fallback pure fluid configuration")
    rng_seed = params["seed"] + index
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    L: float = params["box_size"]
    crystal_positions = np.empty((0, 3))
    n_fluid = N_PARTICLES
    fluid_positions = generate_disordered_particles(n_fluid, crystal_positions, L)
    positions = fluid_positions.copy()
    is_crystal = np.zeros(fluid_positions.shape[0], dtype=int)
    positions = mc_displace_fluid(positions, is_crystal, L, MC_STEPS_PER_PARTICLE, STEP_SIZE)
    save_configuration(index, positions, is_crystal, L, "Fallback pure fluid")


# -----------------------------------------------------------------------------
#  DRIVER  ─────────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------


def main() -> None:
    # Compute cubic box edge length from packing fraction
    v_particle = (math.pi * D ** 3) / 6.0
    box_size = ((N_PARTICLES * v_particle) / ETA) ** (1 / 3)

    print(f"Generating {NUM_SAMPLES} configurations in a box of size L = {box_size:.3f} σ")

    params = {
        "box_size": box_size,
        "num_samples": NUM_SAMPLES,
        "seed": RNG_BASE_SEED,
    }

    n_cores = min(multiprocessing.cpu_count(), 5)
    print(f"Using {n_cores} CPU core(s)")

    worker = partial(generate_configuration, params=params)
    with multiprocessing.Pool(processes=n_cores) as pool:
        list(tqdm(pool.imap(worker, range(NUM_SAMPLES)),
                  total=NUM_SAMPLES, desc="Config"))

    print("Ensemble generation complete – files written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
