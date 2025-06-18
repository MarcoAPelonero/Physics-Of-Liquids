from Network.geomArch import AmorphousParticleGNN
from Network.head import ParticleClassifier
from Training.dataImport2 import create_data_loaders
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import os

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def plot_3d_scatter(ax, pos, values, title="", cmap="viridis", s=5):
    """
    pos    : (N, 3) array of xyz coordinates
    values : (N,)   colour values
    """
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                    c=values, s=s, cmap=cmap)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    return sc

def main():
    # Configuration
    config = {
        'hidden_dim': 128,
        'num_layers': 4,
        'proj_dim': 128,
        'box_size': 1.0,
        'num_epochs': 50,
        'batch_size': 1,
        'lr': 0.0005,
        'k': 6  # Using fixed k for inference
    }
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained GNN
    gnn = AmorphousParticleGNN(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        proj_dim=config['proj_dim'],
        box_size=config['box_size']
    ).to(device)
    
    gnn.load_state_dict(torch.load(r"C:\Users\marcu\Desktop\github repos\Physics-Of-Liquids\FinalProject\ProjectGNN\Saves\best_amorphous_model.pth"))
    gnn = gnn.to(device)  # Ensure all buffers are on the correct device
    # Freeze GNN parameters
    for param in gnn.parameters():
        param.requires_grad = False
    
    # Create classifier head
    classifier = ParticleClassifier(gnn, hidden_dim=128).to(device)
    
    # Load datasets
    data_directory = r"C:\Users\marcu\Desktop\github repos\Physics-Of-Liquids\FinalProject\ProjectGNN\Dataset\hard_sphere_ensemble"
    train_loader, val_loader, test_loader = create_data_loaders(
        data_directory,
        batch_size=config['batch_size'], 
        portion=1, 
        corrupted=True,
        labeled=True,  # This is the key parameter
        skip_files = 250
    )
    # print how many batches are in the train_loader
    print(f"Number of batches in train_loader: {len(train_loader)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get JUST one batch and pass it trough the classifier
    next_batch = next(iter(train_loader))
    if isinstance(next_batch, (list, tuple)):
        pos = next_batch[0]
    else:
        pos = next_batch
    
    pos = pos.to(device)
    
    # Flatten positions and create batch indices
    pos_flat = pos.view(-1, 3)  # [total_particles, 3]
    batch_idx = torch.arange(pos.size(0), device=device).repeat_interleave(pos.size(1))
    print(batch_idx.shape, pos_flat.shape)
    # Pass through classifier
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(pos_flat, batch=batch_idx, k=config['k'])
    
    print(outputs.cpu().numpy())

    # Reshape outputs to match original positions
    outputs = outputs.view(pos.size(0), pos.size(1))  # [batch_size, num_particles]
    print(outputs.shape)

    # Training loop
    num_epochs = config['num_epochs']
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config['lr'])
    # Get total number of positive and negative labels
    total_positive_labels = 0
    total_negative_labels = 0
    for batch in train_loader:
        if isinstance(batch, (list, tuple)):
            pos, labels = batch
        else:
            pos = batch
            labels = None
        if labels is not None:
            labels = labels.to(device).view(-1)
            total_positive_labels += (labels == 1).sum().item()
            total_negative_labels += (labels == 0).sum().item()
        
    print(f"Total positive labels: {total_positive_labels}, Total negative labels: {total_negative_labels}")


    num_pos = total_positive_labels
    num_neg = total_negative_labels

    # 1-B.  Weight = (# negatives / # positives)
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)  # Use Focal Loss
    # Defiune accuracy metric
    def accuracy(preds, labels):
        preds = torch.sigmoid(preds) > 0.5  # Convert logits to binary predictions
        return (preds == labels).float().mean().item()

    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0.0
        total_acc = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if isinstance(batch, (list, tuple)):
                pos, labels = batch
            else:
                pos = batch
            
            pos = pos.to(device)
            batch_idx = torch.arange(pos.size(0), device=device).repeat_interleave(pos.size(1))
            
            optimizer.zero_grad()
            outputs = classifier(pos.view(-1, 3), batch=batch_idx, k=config['k'])
            labels = labels.to(device).view(-1)  # Flatten labels
            
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # Calculate accuracy
            acc = accuracy(outputs, labels)
            total_acc += acc
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {total_acc/len(train_loader):.4f}")

    test_batch = next(iter(test_loader))
    if isinstance(test_batch, (list, tuple)):
        test_pos, test_labels = test_batch          # (B, P, 3) and (B, P)
    else:
        test_pos = test_batch                       # (B, P, 3)
        test_labels = None

    test_pos = test_pos.to(device)
    batch_size, num_particles, _ = test_pos.shape

    # -----  forward pass --------------------------------------------------------
    test_batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_particles)
    with torch.no_grad():
        test_outputs = classifier(test_pos.view(-1, 3),
                                batch=test_batch_idx,
                                k=config["k"])                  # (B·P,)

    # restore (B, P) so it matches test_pos
    test_outputs = test_outputs.view(batch_size, num_particles)

    # -----  flatten for plotting ------------------------------------------------
    pos_flat     = test_pos.view(-1, 3).cpu().numpy()            # (B·P, 3)
    out_flat     = test_outputs.view(-1).cpu().numpy()           # (B·P,)
    
    # Apply sigmoid and threshold to get binary predictions
    out_binary   = (torch.sigmoid(test_outputs.view(-1)) > 0.5).float().cpu().numpy()  # (B·P,)
    
    label_flat   = test_labels.view(-1).cpu().numpy() if test_labels is not None else None

    # -----  plot ----------------------------------------------------------------
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    sc1 = plot_3d_scatter(ax1, pos_flat, out_binary,
                        title="Classifier Predictions (0/1)", cmap="viridis")
    plt.colorbar(sc1, ax=ax1, label="Predicted class")

    ax2 = fig.add_subplot(122, projection="3d")
    sc2 = plot_3d_scatter(ax2, pos_flat, label_flat,
                        title="True Labels", cmap="viridis")
    plt.colorbar(sc2, ax=ax2, label="True label")

    plt.suptitle("Classifier predictions vs. true labels")
    plt.tight_layout()
    plt.show()

    # Print overall accuracy over both validation and test sets
    if test_labels is not None:
        overall_acc = accuracy(test_outputs, test_labels.to(device).view(-1))
        print(f"Overall accuracy on test set: {overall_acc:.4f}")
    else:
        print("No labels provided for test set, skipping accuracy calculation.")


if __name__ == "__main__":
    main()