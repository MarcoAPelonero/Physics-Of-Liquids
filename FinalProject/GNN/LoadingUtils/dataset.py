import torch
from torch.utils.data import Dataset
from typing import List
import os
from glob import glob

class CrystalDataset(Dataset):
    def __init__(self, file_paths: List[str]):
        self.data = []
        self.labels = [] 
        self.box_sizes = []
        
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            box_size = float(lines[0].split(':')[1].strip())
            
            self.box_sizes.append(box_size)

            positions = []
            features = []

            labels = []

            for line in lines[3:]:  
                parts = line.strip().split(',')
                if len(parts) >= 3:  
                    x = float(parts[0]) / box_size  
                    y = float(parts[1]) / box_size
                    z = float(parts[2]) / box_size
                    positions.append([x, y, z])

                    q6 = float(parts[4]) 
                    q10 = float(parts[5]) 
                    features.append([q6, q10])

                    label = int(parts[3])
                    labels.append(label)
            
            positions = torch.tensor(positions, dtype=torch.float32)
            features = torch.tensor(features, dtype=torch.float32)
            node_data = torch.cat([positions, features], dim=1)
            labels = torch.tensor(labels)

            self.data.append(node_data)
            self.labels.append(labels)  

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx] 
    
    def get_box_size(self, idx):
        return self.box_sizes[idx]

def import_datasets(dataset_path: str = None, partitions: List = [0.8, 0.1, 0.1]) -> CrystalDataset:
    """
    Import dataset from the specified path.
    
    Args:
        dataset_path (str): Path to the dataset directory.
        corrupted (bool): If True, force each data sample to shape [900, 3].
    
    Returns:
        CrystalDataset: The loaded dataset.
    """
    if dataset_path is None:
        dataset_path = r'C:\Users\marcu\Desktop\github repos\Physics-Of-Liquids\FinalProject\GNN\Dataset\hard_sphere_ensemble'
        # check if the dataset path exists
        if not os.path.exists(dataset_path):
            dataset_path = r'C:\Users\marcu\OneDrive\Desktop\Stuff\github_repos\Physics-Of-Liquids\FinalProject\GNN\Dataset\hard_sphere_ensemble'

    assert sum(partitions) == 1, "Partitions must sum to 1"
    assert len(partitions) == 3, "Partitions must be a list of three values for train, validation, and test sets"

    file_paths = glob(os.path.join(dataset_path, '*.csv'))

    num_files = len(file_paths)
    train_size = int(num_files * partitions[0])
    val_size = int(num_files * partitions[1])
    test_size = num_files - train_size - val_size

    train_files = file_paths[:train_size]
    val_files = file_paths[train_size:train_size + val_size]
    test_files = file_paths[train_size + val_size:]

    train_dataset = CrystalDataset(train_files)
    val_dataset = CrystalDataset(val_files)
    test_dataset = CrystalDataset(test_files)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")   
    print(f"Test dataset size: {len(test_dataset)}")

    return  train_dataset, val_dataset, test_dataset

if __name__ == "__main__":

    data_folder = r'C:\Users\marcu\Desktop\github repos\Physics-Of-Liquids\FinalProject\GNN\Dataset\hard_sphere_ensemble'
    file_paths = glob(os.path.join(data_folder, '*.csv'))
    dataset = CrystalDataset(file_paths)
    print(f"Dataset size: {len(dataset)}")
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}")

    import random
    import matplotlib.pyplot as plt
    random_index = random.randint(0, len(dataset) - 1)
    sample_data, sample_labels = dataset[random_index]
    
    sample_positions = sample_data[:, :3].numpy()
    sample_features = sample_data[:, 3:].numpy()
    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')

    ax.scatter(sample_positions[:, 0], sample_positions[:, 1], sample_positions[:, 2], c=sample_labels.numpy(), cmap='coolwarm', s=5, alpha=0.7)
    ax.set_title('Labels')
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(sample_positions[:, 0], sample_positions[:, 1], sample_positions[:, 2], c=sample_features[:, 0], cmap='coolwarm', s=5, alpha=0.7)
    ax.set_title('Q6')  
    ax = fig.add_subplot(133, projection='3d')
    ax.scatter(sample_positions[:, 0], sample_positions[:, 1], sample_positions[:, 2], c=sample_features[:, 1], cmap='coolwarm', s=5, alpha=0.7)
    ax.set_title('Q10')
    plt.show()