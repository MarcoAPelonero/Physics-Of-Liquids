import torch
import torch.nn as nn
try:
    from geomArch import AmorphousParticleGNN
except ImportError:
    from Network.geomArch import AmorphousParticleGNN

class ParticleClassifier(nn.Module):
    def __init__(self, gnn: AmorphousParticleGNN, hidden_dim=128):
        super().__init__()
        self.gnn = gnn
        proj_dim = gnn.projection_head[-1].out_features
        
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Fixed output size
        )

    def forward(self, pos, batch=None, k=None):
    # Ensure batch is properly constructed if None
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        
        # Ensure k is properly set
        if k is None:
            k = self.gnn.default_k
        
        particle_emb = self.gnn(pos, batch, k, mode="inference")
        return self.classifier(particle_emb).squeeze(-1)

if __name__ == "__main__":
    config = {
        'hidden_dim':128,
        'num_layers': 4,
        'proj_dim': 128,
        'box_size': 1.0,
        'k_list': [3, 15],       # Local and global k-values
        'temperature': 0.1,      # NT-Xent temperature
        'num_epochs': 40,
        'batch_size': 24,        # Should match your dataloader
        'lr': 0.001,
        'save_path': "best_amorphous_model.pth"
    }
    
    # Initialize model
    gnn = AmorphousParticleGNN(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        proj_dim=config['proj_dim'],
        box_size=config['box_size']
    )
    classifier = ParticleClassifier(gnn).to('cuda')

    # Process 500 particles (single system)
    pos = torch.rand(500, 3).to('cuda')
    particle_probs = classifier(pos, k=10)  # [500]

    print(particle_probs.shape)  # Should be [500]
    print(particle_probs[:10])  # Print first 10 probabilities