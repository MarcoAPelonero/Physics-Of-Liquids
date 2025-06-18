import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph, MessagePassing
from torch_geometric.utils import remove_self_loops

class ParticleLayer(MessagePassing):
    """Single message‑passing block used in the GNN."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + 4, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=1))

    def update(self, aggr_out, x):
        return x + self.update_mlp(torch.cat([x, aggr_out], dim=1))


class AmorphousParticleGNN(nn.Module):
    """GNN for disordered particle systems with periodic boundaries."""
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 10,
        default_k: int = 30,
        proj_dim: int = 128,
        box_size: float | torch.Tensor = 1.0,
    ) -> None:
        super().__init__()
        self.default_k = default_k
        
        # Handle box size (scalar or vector)
        if isinstance(box_size, torch.Tensor):
            self.register_buffer("box", box_size.float())
        else:
            self.register_buffer("box", torch.tensor([box_size] * 3, dtype=torch.float32))

        # Precompute 27 lattice shifts and central index
        shift_vals = torch.tensor([-1, 0, 1], dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(shift_vals, shift_vals, shift_vals, indexing="ij"), dim=-1)
        self.register_buffer("shifts", grid.reshape(-1, 3))
        self.central_shift_idx = torch.argwhere((self.shifts == 0).all(dim=1)).squeeze().item()

        # Network components
        self.encoder = nn.Linear(3, hidden_dim)
        self.layers = nn.ModuleList(
            ParticleLayer(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, pos, batch=None, k=None, mode: str = "inference"):
        if mode == "inference":
            k = k or self.default_k
            edge_index, edge_attr = self._build_pbc_knn(pos, batch, k)
            h = self.encoder(pos)
            for layer in self.layers:
                h = layer(h, edge_index, edge_attr)
            return self.projection_head(h)

        elif mode == "contrastive":
            assert isinstance(k, (list, tuple)) and len(k) == 2, "Need two k values"
            h0 = self.encoder(pos)
            projections = []
            for k_val in k:
                edge_index, edge_attr = self._build_pbc_knn(pos, batch, k_val)
                h = h0
                for layer in self.layers:
                    h = layer(h, edge_index, edge_attr)
                projections.append(self.projection_head(h))
            return projections

        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _build_pbc_knn(self, pos, batch, k):
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        
        # Group by batch
        unique_batches = torch.unique(batch)
        all_edge_index = []
        all_edge_attr = []
        
        for b in unique_batches:
            mask = (batch == b)
            pos_b = pos[mask]  # [N, 3] particles in this batch
            
            # Compute pairwise displacement with PBC
            disp = pos_b.unsqueeze(1) - pos_b.unsqueeze(0)  # [N, N, 3]
            disp = disp - torch.round(disp / self.box) * self.box  # MIC
            
            # Get distances
            dist = torch.norm(disp, dim=-1)  # [N, N]
            
            # Find k-nearest neighbors (excluding self)
            topk_dist, topk_idx = torch.topk(dist, k=k+1, largest=False)
            topk_idx = topk_idx[:, 1:]  # Remove self-loop
            topk_dist = topk_dist[:, 1:]
            
            # Create edges
            src = torch.arange(pos_b.shape[0], device=pos.device).repeat_interleave(k)
            dst = topk_idx.flatten()
            
            # Edge attributes
            edge_disp = disp[src, dst]  # [k*N, 3]
            edge_attr = torch.cat([edge_disp, topk_dist.flatten().unsqueeze(1)], dim=1)
            
            # Offset indices for global batch
            offset = mask.nonzero()[0].min()
            all_edge_index.append(torch.stack([src + offset, dst + offset]))
            all_edge_attr.append(edge_attr)
        
        edge_index = torch.cat(all_edge_index, dim=1)
        edge_attr = torch.cat(all_edge_attr, dim=0)
        return edge_index, edge_attr

if __name__ == "__main__":
    import torch

    # Create 32 independent systems of 900 particles each: (32, 900, 3)
    batch_pos = torch.rand(32, 900, 3)  # [batch_size, num_particles, 3]

    # Instantiate the model
    model = AmorphousParticleGNN(box_size=10.0)
    # Import torch summary and print model summary (everything on the same device)
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    summary(model, input_size=(900, 3), device=str(device))
    
    # Process each system independently (vectorized)
    outputs = []
    for i in range(batch_pos.size(0)):
        # Get positions for system i: (900, 3)
        pos = batch_pos[i]
        
        # Forward pass (no batch vector needed since it's a single system)
        out = model(pos, batch=None, k=30)  # k=30 neighbors
        
        outputs.append(out)

    # Stack results: (32, 900, proj_dim)
    final_output = torch.stack(outputs, dim=0)
    print("Output shape:", final_output.shape)