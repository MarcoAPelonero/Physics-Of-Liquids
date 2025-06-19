import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import Optional, Tuple


class ContrastiveComplexGNN(nn.Module):
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

    def get_contrastive_edges(
        self,
        positions: torch.Tensor,
        q_values: torch.Tensor,
        k_spatial: int,
        k_q: int,
        box_length: float = 1.0,
        similar_q: bool = True
    ) -> torch.Tensor:
        """Build edges considering both spatial proximity and q-value similarity/dissimilarity.
        
        Parameters
        ----------
        positions : torch.Tensor
            Node coordinates of shape (B, N, 3)
        q_values : torch.Tensor
            Node features of shape (B, N, 2) (two q-values per node)
        k_spatial : int
            Number of spatial neighbors to consider
        k_q : int
            Number of q-similar/dissimilar neighbors to consider
        box_length : float
            Simulation box length for PBC
        similar_q : bool
            Whether to connect similar (True) or dissimilar (False) q-values
            
        Returns
        -------
        torch.Tensor
            Edge indices of shape (2, E)
        """
        batch_size, num_nodes, _ = positions.shape
        edge_indices = []
        
        for b in range(batch_size):
            pos = positions[b]  # (N, 3)
            q = q_values[b]     # (N, 2)
            
            # 1. Compute spatial distances with PBC
            delta = torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0))  # (N, N, 3)
            delta = torch.where(delta > 0.5 * box_length, box_length - delta, delta)
            spatial_dist = torch.sqrt((delta**2).sum(-1) + 1e-12)  # (N, N)
            
            # 2. Compute q-value distances (cosine similarity)
            q_dist = 1 - F.cosine_similarity(
                q.unsqueeze(1),  # (N, 1, 2)
                q.unsqueeze(0),  # (1, N, 2)
                dim=-1
            )  # (N, N)
            
            # 3. Combine criteria
            if similar_q:
                combined_score = spatial_dist - 0.5 * q_dist  # favors close in space AND similar q
            else:
                combined_score = spatial_dist + 0.5 * q_dist  # favors close in space BUT dissimilar q
            
            # 4. Mask self-edges
            mask = torch.eye(num_nodes, dtype=torch.bool, device=positions.device)
            combined_score = combined_score.masked_fill(mask, float('inf'))
            
            # 5. Get top-k neighbors for each node
            k_total = min(k_spatial + k_q, num_nodes - 1)  # ensure we don't request more than available
            _, topk = torch.topk(combined_score, k=k_total, largest=False)
            
            # 6. Build edge list
            src = torch.repeat_interleave(torch.arange(num_nodes, device=positions.device), k_total)
            dst = topk.flatten()
            edges = torch.stack([src, dst])
            
            # Adjust for batch indexing
            edges += b * num_nodes
            edge_indices.append(edges)
        
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
        q_values: Optional[torch.Tensor] = None,
        k_spatial: Optional[int] = None,
        k_q: Optional[int] = None,
        q_similar: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with enhanced edge construction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features (B, N, node_dim)
        positions : torch.Tensor
            Node coordinates (B, N, 3)
        q_values : torch.Tensor, optional
            Q-values (B, N, 2), defaults to last two features of x
        k_spatial : int, optional
            Spatial neighbors, defaults to self.k
        k_q : int, optional
            Q neighbors, defaults to self.k//2
        edge_index : torch.Tensor, optional
            Precomputed edges
        """
        if q_values is None:
            q_values = x[..., -2:]  # Assume last two features are q-values
        
        k_spatial = self.k if k_spatial is None else k_spatial
        k_q = max(1, self.k // 2) if k_q is None else k_q
        
        edge_index = self.get_contrastive_edges(
            positions=positions,
            q_values=q_values,
            k_spatial=k_spatial,
            k_q=k_q,
            similar_q=q_similar,
        )
        
        batch_size, num_nodes, _ = x.shape
        h = self.encoder(x.view(-1, x.size(-1)))
        
        for i in range(self.num_message_passes):
            agg = self.message_pass(h, edge_index, self.message_nets[i])
            if self.use_batch_norm:
                agg = self.batch_norms[i](agg)
            h = h + self.update_nets[i](torch.cat([h, agg], dim=1)) if self.use_skip_connections else self.update_nets[i](torch.cat([h, agg], dim=1))
        
        z = self.projection_head(h)
        return z.view(batch_size, num_nodes, -1), h.view(batch_size, num_nodes, -1)