import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from geomArch import AmorphousParticleGNN
# First, let's test your actual network ======================================
print("=== Testing AmorphousParticleGNN ===")

# Create test data
torch.manual_seed(42)
pos = torch.rand(20, 3) * 0.9 + 0.05  # particles inside [0.05, 0.95] box
box_size = 1.0

# Initialize model
model = AmorphousParticleGNN(hidden_dim=64, num_layers=2, k_neighbors=4, box_size=box_size)

# Forward pass
print("\nInput positions shape:", pos.shape)
output = model(pos)
print("Output shape:", output.shape)  # Should be [20, proj_dim]

# Test backward pass
loss = output.sum()
loss.backward()
print("Backward pass completed successfully")

# Print model summary
print("\nModel architecture:")
print(model)

# Now visualize the graph construction =======================================
print("\n=== Visualizing PBC-aware k-NN ===")

def visualize_pbc_knn(pos, k=4, box_size=1.0):
    """Visualize the PBC-aware k-NN graph construction in 2D"""
    pos = pos[:, :2].detach().numpy()  # Use only x,y coordinates for 2D visualization
    n_nodes = len(pos)
    
    # Generate shifts for PBC
    shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 2)
    
    # Create ghost positions
    ghost_pos = pos.reshape(1, -1, 2) + shifts.reshape(-1, 1, 2) * box_size
    ghost_pos = ghost_pos.reshape(-1, 2)
    
    # Build k-NN graph (using scipy's KDTree for simplicity)
    from scipy.spatial import KDTree
    tree = KDTree(ghost_pos)
    distances, indices = tree.query(pos, k=k+1)  # k+1 because point itself is included
    
    # Create graph
    G = nx.Graph()
    
    # Add original nodes
    for i, (x, y) in enumerate(pos):
        G.add_node(i, pos=(x, y), original=True)
    
    # Add edges with periodic connections
    periodic_edges = []
    normal_edges = []
    for i in range(n_nodes):
        for j in indices[i, 1:]:  # skip self
            original_target = j % n_nodes
            shift = shifts[j // n_nodes]
            if np.any(shift != 0):
                periodic_edges.append((i, original_target, shift))
            else:
                normal_edges.append((i, original_target))
            G.add_edge(i, original_target, shift=shift)
    
    # Draw the graph
    plt.figure(figsize=(10, 10))
    
    # Draw the central box
    ax = plt.gca()
    ax.add_patch(Rectangle((0, 0), box_size, box_size, fill=False, edgecolor='red', linewidth=2))
    
    # Draw ghost boxes
    for shift in shifts:
        if np.all(shift == 0):
            continue
        offset = shift * box_size
        ax.add_patch(Rectangle(offset, box_size, box_size, 
                             fill=False, edgecolor='gray', linestyle='--', alpha=0.5))
    
    # Get node positions for drawing
    pos_dict = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos_dict, node_size=100, node_color='blue')
    
    # Draw normal edges first
    nx.draw_networkx_edges(G, pos_dict, edgelist=normal_edges, width=1.5)
    
    # Draw periodic edges with arrows
    for u, v, shift in periodic_edges:
        plt.annotate("",
                    xy=pos_dict[v], xycoords='data',
                    xytext=pos_dict[u], textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--",
                                   connectionstyle=f"arc3,rad=0.2",
                                   color="green", alpha=0.7))
        # Show shift vector
        mid_x = (pos_dict[u][0] + pos_dict[v][0])/2
        mid_y = (pos_dict[u][1] + pos_dict[v][1])/2
        plt.text(mid_x, mid_y, f"{shift}", color='red', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.title(f"PBC-aware {k}-NN graph (2D visualization)\nRed box: central cell, Gray boxes: ghost cells")
    plt.xlim(-box_size, 2*box_size)
    plt.ylim(-box_size, 2*box_size)
    plt.gca().set_aspect('equal')
    plt.show()

# Visualize with same parameters as model
visualize_pbc_knn(pos, k=model.k, box_size=box_size)

# Finally, let's verify the edge attributes ===================================
print("\n=== Verifying edge attributes ===")
edge_index, edge_attr = model.build_pbc_knn(pos, batch=None)
print("Edge index shape:", edge_index.shape)  # Should be [2, num_edges]
print("Edge attr shape:", edge_attr.shape)    # Should be [num_edges, 4] (3 coords + distance)

# Verify periodic wrapping
print("\nExample edge attributes (first 5 edges):")
for i in range(5):
    src, dst = edge_index[:, i]
    attr = edge_attr[i]
    print(f"Edge {i}: {src.item()} -> {dst.item()} | vec={attr[:3].numpy().round(3)}, dist={attr[3].item():.3f}")

# Verify minimum image convention
print("\nTesting minimum image convention:")
test_pos = torch.tensor([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
edge_index, edge_attr = model.build_pbc_knn(test_pos, batch=None)
print("Vector between particles:", edge_attr[0, :3])  # Should be ~[-0.2, -0.2, -0.2] not [0.8, 0.8, 0.8]