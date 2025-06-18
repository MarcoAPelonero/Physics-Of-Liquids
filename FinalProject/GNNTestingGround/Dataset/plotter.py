import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random

def read_crystal_data(filepath):
    """Read crystal data, checking for clusters and valid markers"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Check for cluster info (must exist and not be None)
    if len(lines) < 2 or not lines[1].startswith('# Clusters:') or 'None' in lines[1]:
        return None
    
    box_size = float(lines[0].split(': ')[1].strip())
    cluster_info = lines[1].strip()
    
    # Load data (skip 3 lines: metadata + clusters + header)
    data = np.genfromtxt(filepath, delimiter=',', skip_header=3)
    if data.size == 0:  # Skip empty files
        return None
    
    x, y, z, markers = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    
    # Only keep files where at least one marker == 1
    if 1 not in markers:
        return None
    
    return {
        'box_size': box_size,
        'cluster_info': cluster_info,
        'x': x, 'y': y, 'z': z,
        'markers': markers,
        'filename': os.path.basename(filepath)
    }

def plot_3d_scatter(ax, data):
    """Plot 3D scatter with cluster info"""
    sc = ax.scatter(data['x'], data['y'], data['z'], 
                   c=data['markers'], cmap='viridis', alpha=0.6, s=20)
    ax.set_title(data['filename'])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.text2D(0.05, 0.95, data['cluster_info'], 
             transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    return sc

def main():
    data_dir = 'hard_sphere_ensemble'
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' not found!")
        return
    
    # Load only valid files (with clusters and marker=1)
    valid_data = []
    for file in [f for f in os.listdir(data_dir) if f.endswith(('.txt', '.csv'))]:
        data = read_crystal_data(os.path.join(data_dir, file))
        if data is not None:
            valid_data.append(data)
    
    if not valid_data:
        print("No valid files found (must have clusters AND marker=1)!")
        return
    
    # Select up to 6 random samples
    selected = random.sample(valid_data, min(6, len(valid_data)))
    
    # Plotting
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Crystal Structures with Valid Clusters ({len(selected)} samples)", fontsize=14)
    
    for i, sample in enumerate(selected):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        sc = plot_3d_scatter(ax, sample)
    
    # Add colorbar if we have plots
    if selected:
        plt.colorbar(sc, ax=fig.get_axes(), shrink=0.6, label='Crystal Marker')
    
    plt.show()

if __name__ == '__main__':
    main()