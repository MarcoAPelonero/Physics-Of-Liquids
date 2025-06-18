import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors

def process_file(filepath, k=3):
    """Process a single file and return data needed for plotting"""
    df = pd.read_csv(filepath, delim_whitespace=True)
    points = df[['X', 'Y', 'Z']].values
    q_values = df['Q_l_atom'].values
    
    # Find k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)  # +1 to exclude self
    distances, indices = nbrs.kneighbors(points)
    
    # Prepare line segments with their average Q values
    lines = []
    avg_qs = []
    
    for i in range(len(points)):
        # Skip the first neighbor which is the point itself
        for neighbor_idx in indices[i][1:]:
            # Store the line segment
            lines.append((points[i], points[neighbor_idx]))
            # Calculate average Q value
            avg_q = (q_values[i] + q_values[neighbor_idx]) / 2
            avg_qs.append(avg_q)
    
    return points, lines, avg_qs

def plot_with_plotly(files, k=3):
    """Create subplots using plotly"""
    # Create subplots
    fig = make_subplots(rows=2, cols=3, 
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
                               [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        subplot_titles=[f"File {i+1}" for i in range(len(files))])
    
    for i, filepath in enumerate(files):
        points, lines, avg_qs = process_file(filepath, k)
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        # Normalize Q values for color and alpha
        avg_qs = np.array(avg_qs)
        norm_qs = (avg_qs - avg_qs.min()) / (avg_qs.max() - avg_qs.min())
        
        # Add points
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=4, color='black'),
            name=f'Points {i+1}',
            showlegend=False
        ), row=row, col=col)
        
        # Add lines with color and alpha based on Q values
        for (start, end), q, norm_q in zip(lines, avg_qs, norm_qs):
            # Use a colormap (blue to red)
            color = f'rgb({int(255*norm_q)}, 0, {int(255*(1-norm_q))})'
            
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode='lines',
                line=dict(color=color, width=2),
                opacity=norm_q*0.7 + 0.3,  # Ensure minimum alpha of 0.3
                showlegend=False
            ), row=row, col=col)
    
    fig.update_layout(height=800, width=1200, title_text="Particle Neighbor Connections")
    fig.show()

def plot_with_matplotlib(files, k=3):
    """Create subplots using matplotlib (alternative)"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    fig = plt.figure(figsize=(18, 12))
    
    for i, filepath in enumerate(files):
        points, lines, avg_qs = process_file(filepath, k)
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        
        # Normalize Q values for color and alpha
        avg_qs = np.array(avg_qs)
        norm = Normalize(vmin=avg_qs.min(), vmax=avg_qs.max())
        cmap = cm.get_cmap('viridis')
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', s=20)
        
        # Plot lines with color and alpha based on Q values
        for (start, end), q in zip(lines, avg_qs):
            color = cmap(norm(q))
            alpha = norm(q)*0.7 + 0.3  # Ensure minimum alpha of 0.3
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                   color=color, alpha=alpha, linewidth=1.5)
        
        ax.set_title(f"File {i+1}")
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=fig.get_axes(), label='Average Q value', shrink=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Assuming files are named file1.txt, file2.txt, ..., file6.txt in the current directory
    files = [f"knn_results/q_l_{i}.txt" for i in range(1, 7)]
    
    # Check which files exist
    existing_files = [f for f in files if os.path.exists(f)]
    
    if len(existing_files) == 0:
        print("No files found. Using example data for demonstration.")
        # In a real scenario, you would handle missing files appropriately
        # For now, we'll proceed with the names anyway
        existing_files = files
    
    # Use plotly for faster rendering (interactive)
    print("Creating plot with plotly...")
    plot_with_plotly(existing_files, k=3)
    
    # Uncomment below to use matplotlib instead
    # print("Creating plot with matplotlib...")
    # plot_with_matplotlib(existing_files, k=3)