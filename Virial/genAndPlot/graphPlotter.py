import json
import math
import networkx as nx
import matplotlib.pyplot as plt

def load_independent_graphs(json_filename):
    """Load independent graphs from the JSON file and convert them to networkx graphs."""
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    nx_graphs = []
    for item in data:
        n = item["n"]
        edges = item["edges"]
        multiplicity = item["multiplicity"]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        nx_graphs.append((G, multiplicity))
    return nx_graphs

def plot_independent_graphs(graphs, max_plots=20):
    total_graphs = len(graphs)
    # If more than max_plots, only plot the first max_plots and add an extra subplot for ellipsis.
    if total_graphs > max_plots:
        graphs_to_plot = graphs[:max_plots]
        extra = True
    else:
        graphs_to_plot = graphs
        extra = False

    n_plots = len(graphs_to_plot) + (1 if extra else 0)
    # Choose an optimal grid (approximately square).
    rows = math.ceil(math.sqrt(n_plots))
    cols = math.ceil(n_plots / rows)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axs = axs.flatten()

    for i, (G, multiplicity) in enumerate(graphs_to_plot):
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=axs[i], with_labels=True, node_size=500, font_size=10)
        axs[i].set_title(f"Diagram {i+1}\nMultiplicity: {multiplicity}")
        axs[i].axis('equal')

    if extra:
        axs[len(graphs_to_plot)].text(0.5, 0.5, '...', fontsize=40,
                                       ha='center', va='center')
        axs[len(graphs_to_plot)].set_title("More graphs exist")
        axs[len(graphs_to_plot)].axis('off')

    for j in range(n_plots, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    json_filename = 'independent_graphs.json'
    graphs = load_independent_graphs(json_filename)
    print(f"Loaded {len(graphs)} independent graphs from {json_filename}.")
    plot_independent_graphs(graphs)
