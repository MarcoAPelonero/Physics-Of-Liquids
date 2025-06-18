import torch
import numpy as np
import matplotlib.pyplot as plt
from Network.geomArch import AmorphousParticleGNN
from Training.dataImport import create_data_loaders

# Configuration (must match training config)
config = {
    'hidden_dim': 128,
    'num_layers': 4,
    'proj_dim': 128,
    'box_size': 1.0,
    'batch_size': 24,
    'k_list': [3, 15]  # Important for the model's KNN operations
}

def load_model(model_path):
    """Load trained model with proper configuration"""
    model = AmorphousParticleGNN(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        proj_dim=config['proj_dim'],
        box_size=config['box_size']
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def plot_comparison(original, corrupted, reconstructed, idx):
    """Plot 2D particle position comparisons"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = ['Original', 'Corrupted', 'Reconstructed']
    data = [original, corrupted, reconstructed]
    colors = ['blue', 'red', 'green']
    
    for i, ax in enumerate(axs):
        ax.scatter(data[i][:, 0], data[i][:, 1], s=10, alpha=0.7, c=colors[i])
        ax.set_title(titles[i])
        ax.set_xlim(0, config['box_size'])
        ax.set_ylim(0, config['box_size'])
        ax.set_aspect('equal')
    
    plt.suptitle(f"Sample {idx+1} Comparison", y=1.02)
    plt.tight_layout()
    plt.savefig(f"inference_sample_{idx+1}.png", bbox_inches='tight')
    plt.show()

def main():
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model("Saves/final_model.pth").to(device)
    
    # Load data
    data_dir = r"C:\Users\marcu\Desktop\github repos\Physics-Of-Liquids\FinalProject\ProjectGNN\Dataset\hard_sphere_ensemble"
    _, _, test_loader = create_data_loaders(data_dir, corrupted=True, batch_size=1)  # Use batch_size=1 for easier inference
    
    # Select random test samples
    num_samples = 1
    dataset_size = len(test_loader.dataset)
    indices = np.random.choice(dataset_size, min(num_samples, dataset_size), replace=False)
    
    # Inference loop
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = test_loader.dataset[idx]
            # Remove unsqueeze to get shape [N, 3] for pos
            if isinstance(sample, dict):
                corrupted = sample['corrupted_pos'].to(device)  # shape: [N, 3]
                original = sample['original_pos'].cpu().numpy()
            elif isinstance(sample, torch.Tensor) and sample.dim() == 2:
                corrupted = sample.to(device)  # shape: [N, 3]
                original = sample.cpu().numpy()
            else:
                raise ValueError("Unexpected sample format from dataset.")
            
            # Create batch tensor for each particle
            batch_tensor = torch.zeros(corrupted.shape[0], dtype=torch.long, device=device)
            
            # Forward pass with k-value (using the global k from config)
            k = config['k_list'][1]  # Using the global k-value (15)
            reconstructed = model(corrupted, batch=batch_tensor, k=k, mode="inference")
            
            # Convert to numpy (assuming reconstructed is [N, 3])
            corrupted_np = corrupted.cpu().numpy()
            reconstructed_np = reconstructed.cpu().numpy()
            
            # Plot
            plot_comparison(original, corrupted_np, reconstructed_np, i)

if __name__ == "__main__":
    main()