import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import Optional, Tuple

'''
WARNING:

This network does not handle batching in the most efficient way.
It is designed for single configurations, so it may not scale well to larger datasets.
'''

class ContrastiveGNN(nn.Module):
    """Graph neural network backbone for contrastive pre‑training.

    The network produces two distinct outputs per node:
    1. **Node features** (`h`): High‑dimensional representations intended for
       downstream supervised heads (e.g. classification/regression).  These
       vectors preserve as much task‑relevant information as possible.
    2. **Projection vectors** (`z`): A non‑linear projection of `h` onto a
       typically higher‑dimensional space used exclusively for contrastive
       losses (e.g. InfoNCE).

    🔑 **Tip** When attaching a task‑specific head (after pre‑training) you
    should *only* feed the node features `h` into that head.  The projection
    `z` is optimised for the contrastive objective and is *not* meant for
    direct classification.

    Parameters
    ----------
    node_dim : int, default=5
        Dimensionality of the raw input features per node.
    hidden_dim : int, default=128
        Hidden size used throughout the encoder and message passing stacks.
    proj_dim : int, default=256
        Output dimension of the projection head (`z`).
    k : int, default=5
        Number of nearest neighbours (k‑NN) to connect for each node when
        building the graph on the fly.
    num_encoder_layers : int, default=2
        Depth of the MLP encoder applied to each node before message passing.
    num_message_passes : int, default=3
        Number of message‑passing rounds.
    use_attention : bool, default=False
        If *True*, messages are re‑weighted with dot‑product attention.
    use_skip_connections : bool, default=True
        Enables residual connections between message‑passing layers.
    use_batch_norm : bool, default=False
        If *True*, inserts BatchNorm1d after each aggregation step.
    """

    def __init__(
        self,
        node_dim: int = 5,
        hidden_dim: int = 128,
        proj_dim: int = 256,
        k: int = 5,
        num_encoder_layers: int = 2,
        num_message_passes: int = 3,
        use_attention: bool = False,
        use_skip_connections: bool = True,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.k = k
        self.num_message_passes = num_message_passes
        self.use_attention = use_attention
        self.use_skip_connections = use_skip_connections
        self.use_batch_norm = use_batch_norm
        self.hidden_dim = hidden_dim

        encoder_layers = [nn.Linear(node_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_encoder_layers - 1):
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*encoder_layers)

        self.message_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(num_message_passes)
        ])
        self.update_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(num_message_passes)
        ])
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_message_passes)
            ])

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def get_edges(
        self,
        positions: torch.Tensor,
        k: Optional[int] = None,
        box_length: float = 1.0,
    ) -> torch.Tensor:
        """Build a *periodic* k‑NN graph for each batch element.

        The node coordinates lie in a cubic box of side‐length ``box_length``
        with periodic boundary conditions (PBC).  Distances therefore wrap
        around the edges of the box.  For each node we connect to the *k*
        nearest distinct neighbours.

        Parameters
        ----------
        positions : torch.Tensor
            Cartesian coordinates of shape *(B, N, 3)*.
        k : int | None, default=None
            Number of neighbours.  Uses the value passed to ``__init__`` when
            *None*.
        box_length : float, default=1.0
            Edge length of the PBC simulation box.

        Returns
        -------
        torch.Tensor
            Edge index in *PyG* style of shape *(2, E)* with **global** node
            indices in the flattened *(B × N)* space.
        """
        k = self.k if k is None else k
        batch_size, num_nodes, _ = positions.shape
        edge_indices = []

        for b in range(batch_size):
            pos = positions[b]
            delta = torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0))
            delta = torch.where(delta > 0.5 * box_length, box_length - delta, delta)
            dist = torch.sqrt(delta.pow(2).sum(-1) + 1e-12)
            _, topk = torch.topk(dist, k=k + 1, largest=False)

            src, dst = [], []
            for i in range(num_nodes):
                for j in topk[i][1:]:  # skip self
                    src.append(i)
                    dst.append(j)

            edge = torch.stack([
                torch.tensor(src, device=positions.device) + b * num_nodes,
                torch.tensor(dst, device=positions.device) + b * num_nodes,
            ])
            edge_indices.append(edge)

        return torch.cat(edge_indices, dim=1)

    def message_pass(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        message_net: nn.Module,
    ) -> torch.Tensor:
        """Single round of message passing.

        Messages are computed *per edge* by concatenating sender and receiver
        node features and feeding them through a small MLP.  Optionally, they
        are scaled with dot‑product attention.  Finally, messages are **mean
        aggregated** at the destination nodes.

        Parameters
        ----------
        h : torch.Tensor
            Flattened node features of shape *(B × N, hidden_dim)*.
        edge_index : torch.Tensor
            Edge list as returned by :py:meth:`get_edges`.
        message_net : nn.Module
            Edge MLP for this layer.

        Returns
        -------
        torch.Tensor
            Aggregated messages per node with the same shape as ``h``.
        """
        if edge_index.size(1) == 0:
            return torch.zeros_like(h)

        row, col = edge_index  # senders → receivers
        senders, receivers = h[row], h[col]

        if self.use_attention:
            attn = torch.einsum("ij,ij->i", senders, receivers)
            attn = F.softmax(attn, dim=0).unsqueeze(1)
            msg = message_net(torch.cat([senders, receivers], dim=1)) * attn
        else:
            msg = message_net(torch.cat([senders, receivers], dim=1))

        agg = torch.zeros_like(h)
        agg.index_add_(0, col, msg)

        deg = torch.zeros(h.size(0), device=h.device)
        deg.index_add_(0, col, torch.ones_like(col, dtype=torch.float))
        deg.clamp_(min=1.0)

        return agg / deg.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of graphs.

        Parameters
        ----------
        x : torch.Tensor
            Raw node features of shape *(B, N, node_dim)*.
        positions : torch.Tensor
            Node coordinates used to construct the dynamic graph, shape
            *(B, N, 3)*.
        k : int | None, default=None
            Overrides the default neighbour count during graph construction.

        Returns
        -------
        z : torch.Tensor
            Projection vectors of shape *(B, N, proj_dim)* – **only** for
            contrastive objectives.
        h : torch.Tensor
            Node features of shape *(B, N, hidden_dim)* – feed these into any
            downstream task head after pre‑training.
        """
        batch_size, num_nodes, _ = x.shape

        h = self.encoder(x.view(-1, x.size(-1)))  # (B×N, hidden_dim)

        edge_index = self.get_edges(positions, k)

        for i in range(self.num_message_passes):
            agg = self.message_pass(h, edge_index, self.message_nets[i])
            if self.use_batch_norm:
                agg = self.batch_norms[i](agg)
            h = h + self.update_nets[i](torch.cat([h, agg], dim=1)) if self.use_skip_connections else self.update_nets[i](torch.cat([h, agg], dim=1))

        z = self.projection_head(h)

        z = z.view(batch_size, num_nodes, -1)
        h = h.view(batch_size, num_nodes, -1)
        return z, h

    
if __name__ == "__main__":

    # 5 particles in a 1x1x1 box
    positions = torch.tensor([
        [0.1, 0.1, 0.1],  # Particle 0
        [0.9, 0.1, 0.1],  # Particle 1
        [0.5, 0.5, 0.5],  # Particle 2
        [0.1, 0.9, 0.1],  # Particle 3
        [0.5, 0.1, 0.9],  # Particle 4
        [0.2, 0.3, 0.7]   # Particle 5
    ], dtype=torch.float32)

    # Two extra features per particle (e.g., mass, charge)
    features = torch.tensor([
        [1.0, 0.5],  # Particle 0
        [0.8, 0.2],  # Particle 1
        [0.3, 0.9],  # Particle 2
        [0.7, 0.1],  # Particle 3
        [0.5, 0.5],  # Particle 4
        [0.8, 0.3]   # Particle 5
    ], dtype=torch.float32)

    node_data = torch.cat([positions, features], dim=1)
    print("Node features:\n", node_data)
    print("Shape of node data:", node_data.shape)  # Should be [6, 5] (6 particles, 5 features)
    node_data = node_data.unsqueeze(0)  # Add batch dimension
    print("Node data after unsqueeze:", node_data.shape)  # Should be [1, 6, 5] (1 batch, 6 particles, 5 features)
    # Simplified model with 1 layer for everything
    model = ContrastiveGNN(
        node_dim=5,          # 3 pos + 2 features
        hidden_dim=4,        # Smaller for demonstration
        proj_dim=3,          # Final projection size
        k=3,                 # 3 neighbors
        num_encoder_layers=1,# Only 1 encoder layer
        num_message_passes=1,# Only 1 message pass
        use_attention=False, # No attention
        use_skip_connections=False, # No skip connections
        use_batch_norm=False # No batch norm
    )

    print("\n=== Model Architecture ===")
    print("Encoder:", model.encoder)
    print("Message net:", model.message_nets[0])
    print("Update net:", model.update_nets[0])
    print("Projection head:", model.projection_head)

    positions = node_data[:, :, :3]  # Just x,y,z coordinates
    print(positions.shape)  # Should be [1, 6, 3] after unsqueeze

    print("\n=== STEP 1: Node Encoding ===")
    # Flatten the batch dimension as done in the forward method
    h_flat = model.encoder(node_data.view(-1, node_data.size(-1)))
    print("Encoded features (flattened):\n", h_flat)
    print("Shape:", h_flat.shape)  # Should be [6, 4]

    print("\n=== STEP 2: Graph Construction ===")
    edge_index = model.get_edges(positions)
    print("Edge connections (sources -> targets):\n", edge_index.t())

    print("\n=== STEP 3: Message Passing ===")
    # Use the FLATTENED features for message passing
    aggregated = model.message_pass(h_flat, edge_index, model.message_nets[0])
    print("Aggregated messages:\n", aggregated)

    print("\n=== STEP 4: Node Update ===")
    # Continue with flattened representation
    h_updated_flat = model.update_nets[0](torch.cat([h_flat, aggregated], dim=1))
    print("Updated node features (flat):\n", h_updated_flat)

    print("\n=== STEP 5: Projection ===")
    z_flat = model.projection_head(h_updated_flat)
    print("Projections (flat):\n", z_flat)

    # Reshape back to batch form
    batch_size = node_data.shape[0]
    num_nodes = node_data.shape[1]
    z = z_flat.view(batch_size, num_nodes, -1)
    h_out = h_updated_flat.view(batch_size, num_nodes, -1)
    print("\nFinal projections (batch form):", z.shape)
    print("Final node features (batch form):", h_out.shape)
    # Now test an actual pass with a real network
    print("\n=== STEP 6: Full Forward Pass ===")
    model = ContrastiveGNN()
    # Random set of particles
    num_particles = 10
    batch_size = 3
    positions = torch.rand(batch_size, num_particles, 3)  # Random positions in 3D
    features = torch.rand(batch_size, num_particles, 2)   # Random features (e.g., mass, charge)
    # The concatenation must be done along the last dimension
    # dim 0 concatenates along the batch, dim 1 along the particle index
    # While dim 2 is the feature dimension
    node_data = torch.cat([positions, features], dim=2)

    print("Node features (particles):\n", node_data.shape)

    z, h = model(node_data, positions)
    print("Final projection shape:", z.shape)  # Should be [batch_size, num_particles, proj_dim]
    print("Node features shape:", h.shape)     # Should be [batch_size, num_particles, hidden_dim]