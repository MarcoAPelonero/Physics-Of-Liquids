from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

def visualize_features(model, dataloader, device, title="Feature Space Visualization"):
    model.eval()
    features = []
    labels_list = []
    model.to(device)
    
    with torch.no_grad():
        for data, labels in dataloader:
            positions = data[:, :, :3]
            data = data.to(device)
            positions = positions.to(device)
            
            # Process labels: convert node-level → graph-level
            labels = labels[:, 0]  # Take first node's label per graph (if homogeneous)
            # OR use: labels = labels.mean(dim=1)  # For continuous labels
            
            _, node_features = model(data, positions)
            graph_features = node_features.mean(dim=1)  # Graph embeddings
            features.append(graph_features)
            labels_list.append(labels)  # Now graph-level labels
    
    features = torch.cat(features, dim=0).cpu().numpy()
    labels = torch.cat(labels_list, dim=0).cpu().numpy()
    
    # Normalize & apply t-SNE
    features = normalize(features, axis=1)  # Use sklearn normalize
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300)
    features_2d = tsne.fit_transform(features)
    
    # Plot (now shapes match)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()

def create_dataloaders(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to load.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: Configured DataLoader instance.
    """

    # Normalize the dataset if needed


    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Use pinned memory for faster data transfer to GPU
    )

# Example usage:
if __name__ == "__main__":
    from dataset import CrystalDataset  # Adjust the import based on your structure
    import os
    from glob import glob

    data_folder = r'C:\Users\marcu\Desktop\github repos\Physics-Of-Liquids\FinalProject\GNN\Dataset\hard_sphere_ensemble'
    file_paths = glob(os.path.join(data_folder, '*.csv'))
    dataset = CrystalDataset(file_paths)

    dataloader = create_dataloaders(dataset, batch_size=32, shuffle=True, num_workers=4)

    for batch in dataloader:
        data, labels = batch
        print(f"Batch data shape: {data.shape}, Labels shape: {labels.shape}")