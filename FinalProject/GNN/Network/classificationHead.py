import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn

class ParticleClassifier(nn.Module):
    """Classification head for per-particle binary prediction.
    
    Takes the node features (h) from ContrastiveGNN and predicts
    a probability for each particle.
    
    Parameters
    ----------
    hidden_dim : int
        Input dimension of the node features (should match ContrastiveGNN's hidden_dim).
    num_layers : int, default=2
        Number of hidden layers in the classifier.
    dropout : float, default=0.1
        Dropout probability for regularization.
    """
    
    def __init__(
        self, 
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass for per-particle classification.
        
        Parameters
        ----------
        h : torch.Tensor
            Node features from ContrastiveGNN with shape [batch_size, num_particles, hidden_dim]
            
        Returns
        -------
        torch.Tensor
            Per-particle predictions with shape [batch_size, num_particles]
            (sigmoid output for binary classification)
        """
        # Pass through classifier network
        logits = self.net(h)  # [batch_size, num_particles, 1]
        
        # Remove last dimension and apply sigmoid
        return logits.squeeze(-1)  # [batch_size, num_particles]
    
class ContrastiveParticleClassifier(nn.Module):
    """End-to-end contrastive GNN with particle classification head."""
    
    def __init__(
        self,
        gnn_backbone,  # Renamed from 'gnn'
        hidden_dim: int = 64,  # Default hidden dimension
        freeze_backbone: bool = True,   # Renamed from 'freeze_gnn'
        **classifier_kwargs
    ) -> None:
        super().__init__()
        self.gnn_backbone = gnn_backbone  # Clear distinction
        self.classifier = ParticleClassifier(
            hidden_dim=gnn_backbone.hidden_dim,  # Use backbone's hidden_dim
            **classifier_kwargs
        )
        
        if freeze_backbone:
            self.freeze_backbone()  # Now calls method explicitly
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        z, h = self.gnn_backbone(x, positions)  # Clear variable name
        return self.classifier(h)
    
    def freeze_backbone(self) -> None:  # Renamed from 'freeze_gnn'
        """Freeze all GNN backbone parameters."""
        for param in self.gnn_backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:  # Renamed from 'unfreeze_gnn'
        """Unfreeze all GNN backbone parameters."""
        for param in self.gnn_backbone.parameters():
            param.requires_grad = True

def test_representations(model, dataloader, device):
    model.to(device)
    model.eval()
    features, labels = [], []
    
    with torch.no_grad():
        for data, label in dataloader:
            positions = data[:, :, :3].to(device)
            data = data.to(device)
            
            _, h = model(data, positions, k=3)

            # Flatten features and labels:
            features.append(h.reshape(-1, h.size(-1)))
            labels.append(label.view(-1))
    
    X = torch.cat(features).cpu().numpy()
    y = torch.cat(labels).cpu().numpy()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = LogisticRegression().fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Linear probe accuracy: {acc:.4f}")
    return acc