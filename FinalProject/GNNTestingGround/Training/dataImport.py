import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
from sklearn.model_selection import train_test_split

class AtomicPositionDataset(Dataset):
    def __init__(self, file_paths: List[str], corrupted: bool = False):
        """
        Args:
            file_paths: List of paths to data files
            corrupted: If True, force each data sample to shape [900, 3]
        """
        self.data = []
        
        for file_path in file_paths:
            # Read and parse each file
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Extract box size
            box_size = float(lines[0].split(':')[1].strip())
            
            # Parse atomic positions
            positions = []
            for line in lines[3:]:  # Skip header lines
                parts = line.strip().split(',')
                if len(parts) >= 3:  # Ensure we have x,y,z
                    x = float(parts[0]) / box_size  # Normalize by box size
                    y = float(parts[1]) / box_size
                    z = float(parts[2]) / box_size
                    positions.append([x, y, z])
            
            pos_tensor = torch.tensor(positions, dtype=torch.float32)
            if corrupted:
                # Enforce shape [900, 3]: truncate if needed, or pad with zeros.
                if pos_tensor.shape[0] >= 900:
                    pos_tensor = pos_tensor[:900, :]
                else:
                    pad_len = 900 - pos_tensor.shape[0]
                    padding = torch.zeros((pad_len, 3), dtype=torch.float32)
                    pos_tensor = torch.cat((pos_tensor, padding), dim=0)
                    
            self.data.append(pos_tensor)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]  # Only return positions

def create_data_loaders(data_dir: str, 
                        portion = 1.0,
                        train_ratio: float = 0.8, 
                        val_ratio: float = 0.11, 
                        test_ratio: float = 0.1, 
                        batch_size: int = 32, 
                        random_seed: int = 42, 
                        corrupted: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test data loaders from directory of data files.
    
    Args:
        data_dir: Directory containing data files
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        batch_size: Batch size for DataLoader
        random_seed: Random seed for reproducibility
        corrupted: If True, force each sample to shape [900, 3]
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    file_paths = glob.glob(os.path.join(data_dir, '*.csv'))  # Adjust extension if needed
    if not file_paths:
        raise ValueError(f"No data files found in {data_dir}")
    
    # First pass to get cluster information for balanced splitting
    has_cluster_flags = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        cluster_line = lines[1].split(':')[1].strip().lower()
        has_cluster_flags.append(cluster_line != 'none')
    
    n_total = len(has_cluster_flags)
    n_cluster = sum(has_cluster_flags)
    n_non_cluster = n_total - n_cluster
    print(f"Total files: {n_total}")
    print(f" - With cluster flag: {n_cluster}")
    print(f" - Without cluster flag: {n_non_cluster}")
    # Create full dataset (without cluster info)
    full_dataset = AtomicPositionDataset(file_paths, corrupted=corrupted)
    
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split while maintaining cluster/non-cluster balance
    cluster_indices = [i for i, flag in enumerate(has_cluster_flags) if flag]
    non_cluster_indices = [i for i, flag in enumerate(has_cluster_flags) if not flag]
    
    np.random.seed(random_seed)
    np.random.shuffle(cluster_indices)
    np.random.shuffle(non_cluster_indices)
    
    cluster_train_size = int(train_ratio * len(cluster_indices))
    cluster_val_size = int(val_ratio * len(cluster_indices))
    
    non_cluster_train_size = int(train_ratio * len(non_cluster_indices))
    non_cluster_val_size = int(val_ratio * len(non_cluster_indices))
    
    cluster_train = cluster_indices[:cluster_train_size]
    cluster_val = cluster_indices[cluster_train_size:cluster_train_size+cluster_val_size]
    cluster_test = cluster_indices[cluster_train_size+cluster_val_size:]
    
    non_cluster_train = non_cluster_indices[:non_cluster_train_size]
    non_cluster_val = non_cluster_indices[non_cluster_train_size:non_cluster_train_size+non_cluster_val_size]
    non_cluster_test = non_cluster_indices[non_cluster_train_size+non_cluster_val_size:]
    
    train_indices = cluster_train + non_cluster_train
    val_indices = cluster_val + non_cluster_val
    test_indices = cluster_test + non_cluster_test
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create data loaders
    def collate_fn(batch):
        # Stack all position tensors in the batch
        return torch.stack(batch)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    data_directory = r"C:\Users\marcu\Desktop\github repos\Physics-Of-Liquids\FinalProject\ProjectGNN\Dataset\hard_sphere_ensemble"
    
    # To enable the corruption flag, simply pass corrupted=True below
    train_loader, val_loader, test_loader = create_data_loaders(data_directory, corrupted=True)
    
    # Test the loaders
    for batch in train_loader:
        print(f"Batch shape: {batch.shape}")  # Should be [batch_size, num_atoms, 3]
        break