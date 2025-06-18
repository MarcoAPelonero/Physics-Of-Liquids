import numpy as np
import os
import csv

def save_configuration(index: int, positions: np.ndarray, is_crystal: np.ndarray,
                      box_size: float, cluster_info: str, q6: np.ndarray, q10: np.ndarray,
                      output_dir: str = 'hard_spheres') -> None:
    """Save configuration with positions, crystal markers, and Steinhardt parameters."""
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"config_{index:03d}.csv")
    # Before writing, mix up the order of the arrays

    order = np.random.permutation(len(positions))
    positions = positions[order]
    is_crystal = is_crystal[order]
    q6 = q6[order]
    q10 = q10[order]

    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# Box size (L): {box_size:.6f}"])
        w.writerow([f"# Clusters: {cluster_info}"])
        w.writerow(["x", "y", "z", "crystal_marker", "q6", "q10"])
        for pos, marker, q6_val, q10_val in zip(positions, is_crystal, q6, q10):
            w.writerow([pos[0], pos[1], pos[2], int(marker), f"{q6_val:.6f}", f"{q10_val:.6f}"])

def read_crystal_data(filepath):
    """Read crystal data, checking for clusters and valid markers"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2 or not lines[1].startswith('"# Clusters:') or 'None' in lines[1]:
        return None
    
    box_size = float(lines[0].split(': ')[1].strip())
    cluster_info = lines[1].strip()
    
    data = np.genfromtxt(filepath, delimiter=',', skip_header=3)
    if data.size == 0: 
        return None
    
    x, y, z, markers = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    q6, q10 = data[:, 4], data[:, 5]
    
    if 1 not in markers:
        return None
    
    return {
        'box_size': box_size,
        'cluster_info': cluster_info,
        'x': x, 'y': y, 'z': z,
        'q6': q6, 'q10': q10,
        'markers': markers,
        'filename': os.path.basename(filepath)
    }