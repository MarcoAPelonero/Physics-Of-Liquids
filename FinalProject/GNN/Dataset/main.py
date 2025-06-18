from math import pi

D: float = 1.0          
ETA: float = 0.1      # Packing fraction ϕ
N_PARTICLES: int = 1_000 # Total particles per configuration

# Crystalline mini‑clusters
NUM_CLUSTERS_MIN: int = 2   # At least one cluster in an *ordered* sample
NUM_CLUSTERS_MAX: int = 4   # Increase/decrease to get sparser or denser seeds
CLUSTER_MIN_PARTICLES: int = 25
CLUSTER_MAX_PARTICLES: int = 60

#  Monte‑Carlo parameters
MC_STEPS_PER_PARTICLE: int = 250   # How long we randomise the *fluid* particles
STEP_SIZE: float = 0.15 * D        # Initial displacement amplitude (adaptive)

#  Noise at the crystal–fluid interface
INSIDE_NOISE_FRACTION: float = 0.10  # Fraction of cluster sites relabelled as fluid

#  Output: how many configs to generate and where to put them
NUM_SAMPLES: int = 600
OUTPUT_DIR: str = "hard_sphere_ensemble"
RNG_BASE_SEED: int = 42

v_particle = (pi * D ** 3) / 6.0
box_size = ((N_PARTICLES * v_particle) / ETA) ** (1 / 3)

from utils.crystalGeneration import try_place_cluster
from utils.liquidDynamics import generate_disordered_particles, mc_displace
from utils.importExport import save_configuration
from utils.orderParam import compute_Ql_knn
import multiprocessing
import numpy as np
from tqdm import tqdm
import random
import os
from functools import partial

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_configuration(index: int, params: dict) -> None:
    max_retries = 10
    tries = 0
    while tries < max_retries:
        try:
            rng_seed = params["seed"] + index
            np.random.seed(rng_seed)
            random.seed(rng_seed)

            L = params["box_size"]
            N_PARTICLES = params["n_particles"]
            CLUSTER_MIN_PARTICLES = params["cluster_min_particles"]
            CLUSTER_MAX_PARTICLES = params["cluster_max_particles"]
            NUM_CLUSTERS_MIN = params["num_clusters_min"]
            NUM_CLUSTERS_MAX = params["num_clusters_max"]
            INSIDE_NOISE_FRACTION = params["inside_noise_fraction"]
            MC_STEPS_PER_PARTICLE = params["mc_steps_per_particle"]
            STEP_SIZE = params["step_size"]
            OUTPUT_DIR = params.get("output_dir", "hard_spheres")

            # Generate only configurations with crystals
            crystal_positions = []
            cluster_descriptions = []
            remaining = N_PARTICLES

            # Place crystal clusters
            n_clusters = random.randint(NUM_CLUSTERS_MIN, NUM_CLUSTERS_MAX)
            for _ in range(n_clusters):
                if remaining < CLUSTER_MIN_PARTICLES:
                    break
                c_type = random.choice(["SC", "BCC", "FCC"])
                c_size = random.randint(
                    CLUSTER_MIN_PARTICLES,
                    min(CLUSTER_MAX_PARTICLES, remaining)
                )
                existing = np.vstack(crystal_positions) if crystal_positions else np.empty((0, 3))
                placed = try_place_cluster(c_type, c_size, L, existing)
                crystal_positions.append(placed)
                cluster_descriptions.append(f"{c_type}:{c_size}")
                remaining -= c_size

            if not crystal_positions:
                raise RuntimeError("Failed to place any crystal clusters")
                
            crystal_positions = np.vstack(crystal_positions)
            description_str = ", ".join(cluster_descriptions)

            # Add structural noise
            n_noise = int(INSIDE_NOISE_FRACTION * crystal_positions.shape[0])
            noise_indices = np.random.choice(
                crystal_positions.shape[0], 
                n_noise, 
                replace=False
            ) if n_noise > 0 else []

            # Generate fluid particles
            fluid_positions = generate_disordered_particles(remaining, fixed_positions=crystal_positions, box_size=L)
            positions = np.vstack([crystal_positions, fluid_positions])
            is_crystal = np.concatenate([
                np.ones(crystal_positions.shape[0], dtype=int),
                np.zeros(fluid_positions.shape[0], dtype=int),
            ])
            if n_noise:
                is_crystal[noise_indices] = 0  # Convert some crystal to fluid

            # Monte Carlo relaxation
            mobile_indices = np.where(is_crystal == 0)[0].tolist()  # Only fluid particles are mobile
            positions = mc_displace(
                positions, 
                mobile_indices, 
                L,
                MC_STEPS_PER_PARTICLE, 
                STEP_SIZE
            )

            # Compute Steinhardt order parameters (q6 and q10) after final positions
            box = np.array([[0, L], [0, L], [0, L]])
            n_neighbors = min(6, positions.shape[0]-1)
            if n_neighbors > 0:
                results = compute_Ql_knn(
                    positions, 
                    l_list=[6, 10],  
                    n_neighbors=n_neighbors, 
                    box=box
                )
                q6 = np.array([item['q_l_atom'] for item in results[6]['atom_indices']])
                q10 = np.array([item['q_l_atom'] for item in results[10]['atom_indices']])
            else:
                q6 = np.zeros(len(positions))
                q10 = np.zeros(len(positions))

            save_configuration(
                index, 
                positions, 
                is_crystal, 
                L, 
                description_str,
                q6,
                q10,
                OUTPUT_DIR
            )
            return

        except RuntimeError as e:
            tries += 1
            print(f"Config {index}: Error: {e}. Retrying ({tries}/{max_retries})")

    print(f"Config {index}: Using fallback (pure fluid)")
    rng_seed = params["seed"] + index
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    L = params["box_size"]
    N_PARTICLES = params["n_particles"]
    OUTPUT_DIR = params.get("output_dir", "hard_spheres")

    positions = generate_disordered_particles(N_PARTICLES, L)
    is_crystal = np.zeros(N_PARTICLES, dtype=int)
    positions = mc_displace(
        positions, 
        is_crystal, 
        L,
        params["mc_steps_per_particle"], 
        params["step_size"]
    )
    
    box = np.array([[0, L], [0, L], [0, L]])
    n_neighbors = min(12, positions.shape[0]-1)
    if n_neighbors > 0:
        results = compute_Ql_knn(
            positions, 
            l_list=[6, 10], 
            n_neighbors=n_neighbors, 
            box=box
        )
        q6 = np.array([item['q_l_atom'] for item in results[6]['atom_indices']])
        q10 = np.array([item['q_l_atom'] for item in results[10]['atom_indices']])
    else:
        q6 = np.zeros(len(positions))
        q10 = np.zeros(len(positions))
    
    save_configuration(
        index, 
        positions, 
        is_crystal, 
        L, 
        "Fallback: pure fluid",
        q6,
        q10,
        OUTPUT_DIR
    )

def main() -> None:
    v_particle = (pi * D ** 3) / 6.0
    box_size = ((N_PARTICLES * v_particle) / ETA) ** (1 / 3)

    print(f"Generating {NUM_SAMPLES} configurations in a box of size L = {box_size:.3f} σ")

    params = {
        "box_size": box_size,
        "n_particles": N_PARTICLES,
        "cluster_min_particles": CLUSTER_MIN_PARTICLES,
        "cluster_max_particles": CLUSTER_MAX_PARTICLES,
        "num_clusters_min": NUM_CLUSTERS_MIN,
        "num_clusters_max": NUM_CLUSTERS_MAX,
        "inside_noise_fraction": INSIDE_NOISE_FRACTION,
        "mc_steps_per_particle": MC_STEPS_PER_PARTICLE,
        "step_size": STEP_SIZE,
        "output_dir": OUTPUT_DIR,
        "seed": RNG_BASE_SEED,
        "num_samples": NUM_SAMPLES,
    }

    n_cores = min(multiprocessing.cpu_count(), 4)
    print(f"Using {n_cores} CPU core(s)")

    worker = partial(generate_configuration, params=params)
    with multiprocessing.Pool(processes=n_cores) as pool:
        list(tqdm(pool.imap(worker, range(NUM_SAMPLES)),
                  total=NUM_SAMPLES, desc="Config"))

    print("Ensemble generation complete – files written to", OUTPUT_DIR)

def plot_generated_configs():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from utils.importExport import read_crystal_data

    n_files = 2
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("config_") and f.endswith(".csv")]
    
    random_files = random.sample(files, n_files)
    print(f"Plotting {n_files} configurations from {OUTPUT_DIR}")

    fig, axs = plt.subplots(1, n_files, subplot_kw={'projection': '3d'}, figsize=(15, 7))
    for ax, fname in zip(axs, random_files):
        filepath = os.path.join(OUTPUT_DIR, fname)
        data = read_crystal_data(filepath)
        if data is None:
            print(f"Skipping file '{fname}' due to invalid data.")
            continue

        positions = np.column_stack((data['x'], data['y'], data['z']))
        markers = data['markers']
        q6 = data['q6']
        q10 = data['q10']

        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c=markers, cmap='coolwarm', s=5, alpha=0.7)
        ax.set_title(f"{data['filename']} - q6: {np.mean(q6):.2f}, q10: {np.mean(q10):.2f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1]) 
    plt.show()

if __name__ == "__main__":
    main()
    plot_generated_configs()