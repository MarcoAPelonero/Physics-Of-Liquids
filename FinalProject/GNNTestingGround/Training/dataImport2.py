# ---------------------------------------------------------------------------
# unified_particle_dataset.py   (PyTorch-only version – no PyG dependency)
# ---------------------------------------------------------------------------
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Optional

# ────────────────────────────────────────────────────────────────────────────
#  Dataset
# ────────────────────────────────────────────────────────────────────────────
class ParticleDataset(Dataset):
    """
    Loads particle-position CSV files.

    • If `labeled=True`, each line must have an extra 4th column with an
      integer per-particle label.  `__getitem__` will return (pos, label).
    • If `labeled=False`, `__getitem__` returns only pos.
    • If `corrupted=True`, every sample is forced to shape (900, 3), padded
      with zeros (and label -1) or truncated as needed.
    """
    def __init__(self,
                 file_paths: List[str],
                 corrupted: bool = False,
                 labeled:   bool = False):
        self.corrupted = corrupted
        self.labeled   = labeled
        self.data:   List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = [] if labeled else None

        for fp in file_paths:
            with open(fp, "r") as f:
                lines = f.readlines()

            box_size = float(lines[0].split(":")[1].strip())
            positions, particle_labels = [], []

            for ln in lines[3:]:
                parts = ln.strip().split(",")
                if len(parts) < (4 if labeled else 3):
                    continue  # skip malformed lines

                x = float(parts[0]) / box_size
                y = float(parts[1]) / box_size
                z = float(parts[2]) / box_size
                positions.append([x, y, z])

                if labeled:
                    try:
                        particle_labels.append(int(parts[3]))
                    except ValueError:
                        particle_labels.append(-1)

            pos_tensor = torch.tensor(positions, dtype=torch.float32)

            if labeled:
                label_tensor = torch.tensor(particle_labels, dtype=torch.long)

            # --- corruption / padding -------------------------------------------------
            if corrupted:
                if pos_tensor.size(0) >= 900:
                    pos_tensor = pos_tensor[:900]
                    if labeled:
                        label_tensor = label_tensor[:900]
                else:
                    pad = 900 - pos_tensor.size(0)
                    pos_tensor = torch.cat(
                        [pos_tensor, torch.zeros(pad, 3)], dim=0
                    )
                    if labeled:
                        label_tensor = torch.cat(
                            [label_tensor, torch.full((pad,), -1, dtype=torch.long)]
                        )

            self.data.append(pos_tensor)
            if labeled:
                self.labels.append(label_tensor)

    # -----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    # -----------------------------------------------------------------------
    def __getitem__(self, idx):
        if self.labeled:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

# ────────────────────────────────────────────────────────────────────────────
#  Data-loader helper
# ────────────────────────────────────────────────────────────────────────────
def create_data_loaders(
    data_dir:      str,
    portion:       float = 1.0,
    train_ratio:   float = 0.80,
    val_ratio:     float = 0.11,
    test_ratio:    float = 0.09,
    batch_size:    int   = 32,
    random_seed:   int   = 42,
    corrupted:     bool  = False,
    labeled:       bool  = False,
    skip_files:    int   = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not file_paths:
        raise ValueError(f"No *.csv files found in {data_dir}")

    # Skip / portion selection ------------------------------------------------
    file_paths = file_paths[skip_files:]
    file_paths = file_paths[: int(len(file_paths) * portion)]
    print(f"Using {len(file_paths)} files (skip={skip_files}, portion={portion})")

    # Build cluster/non-cluster flags from the header line 2 ------------------
    cluster_flags = []
    for fp in file_paths:
        with open(fp, "r") as f:
            lines = f.readlines()
        flag = (len(lines) >= 2 and lines[1].split(":")[1].strip().lower() != "none")
        cluster_flags.append(flag)

    # Dataset object ----------------------------------------------------------
    full_ds = ParticleDataset(
        file_paths, corrupted=corrupted, labeled=labeled
    )

    # Balanced index splitting ------------------------------------------------
    rng = np.random.default_rng(random_seed)
    cluster_idx     = rng.permutation([i for i, f in enumerate(cluster_flags) if f])
    non_cluster_idx = rng.permutation([i for i, f in enumerate(cluster_flags) if not f])

    def split_indices(arr):
        n = len(arr)
        n_train = int(train_ratio * n)
        n_val   = int(val_ratio   * n)
        train = arr[:n_train]
        val   = arr[n_train : n_train + n_val]
        test  = arr[n_train + n_val :]
        return train, val, test

    c_train, c_val, c_test = split_indices(cluster_idx)
    nc_train, nc_val, nc_test = split_indices(non_cluster_idx)

    train_idx = np.concatenate([c_train, nc_train]).tolist()
    val_idx   = np.concatenate([c_val,   nc_val]).tolist()
    test_idx  = np.concatenate([c_test,  nc_test]).tolist()

    # Ensure indices are integers (fixes the TypeError)
    train_idx = [int(x) for x in train_idx]
    val_idx   = [int(x) for x in val_idx]
    test_idx  = [int(x) for x in test_idx]

    # Subsets -----------------------------------------------------------------
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)

    # Collate fns -------------------------------------------------------------
    if labeled:
        def collate_fn(batch):
            pos, lab = zip(*batch)
            return torch.stack(pos), torch.stack(lab)
    else:
        def collate_fn(batch):
            return torch.stack(batch)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    return train_loader, val_loader, test_loader

# ────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data_dir = r"C:\path\to\hard_sphere_ensemble"
    tr, va, te = create_data_loaders(
        data_dir,
        portion   = 0.05,
        corrupted = True,
        labeled   = True,  # or False
        batch_size= 24,
    )
    for pos, lab in tr if tr.dataset.dataset.labeled else [next(iter(tr))]:
        print("pos:", pos.shape, "lab:", lab.shape if tr.dataset.dataset.labeled else None)
        break
