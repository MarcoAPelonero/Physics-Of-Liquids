import torch
import torch.nn.functional as F
from tqdm import tqdm
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_gpu_memory():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU Memory - Used: {info.used/1024**2:.2f}MB | Free: {info.free/1024**2:.2f}MB")

# Insert before critical operations
print_gpu_memory()

def nt_xent_loss(z1, z2, temperature=0.1, chunk_size=512):
    """
    Memory-efficient NT-Xent loss with chunked similarity computation.
    Args:
        z1, z2: Projected embeddings [N, proj_dim]
        chunk_size: Number of rows to process at once
    """
    N = z1.size(0)
    device = z1.device
    
    # Normalize embeddings
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    
    # Initialize loss
    total_loss = torch.tensor(0.0, device=device)
    
    # Process in chunks
    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        current_chunk_size = chunk_end - chunk_start
        
        # Compute chunk similarities
        sim_chunk = torch.mm(
            z1[chunk_start:chunk_end], 
            z2.t()  # Compare against all embeddings
        ) / temperature
        
        # Create labels (diagonal elements are positives)
        labels = torch.arange(
            chunk_start, chunk_end, 
            dtype=torch.long, device=device
        )
        
        # Cross-entropy for this chunk
        chunk_loss = nn.functional.cross_entropy(
            sim_chunk, labels, reduction='sum'
        )
        
        total_loss += chunk_loss
    
    return total_loss / N

def nt_xent_accuracy(z1, z2, temperature=0.1, chunk_size=512):
    """
    Computes contrastive accuracy in chunked manner
    Returns:
        accuracy: Percentage of correct positive pair identifications
    """
    N = z1.size(0)
    device = z1.device
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    total_correct = 0
    
    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        current_chunk_size = chunk_end - chunk_start
        
        # Compute chunk similarities
        sim_chunk = torch.mm(z1[chunk_start:chunk_end], z2.t()) / temperature
        
        # Create labels (diagonal elements are positives)
        labels = torch.arange(chunk_start, chunk_end, device=device)
        
        # Get predictions
        preds = torch.argmax(sim_chunk, dim=1)
        correct = (preds == labels).sum().item()
        total_correct += correct
        
    return total_correct / N

def train_step(model, pos, optimizer, k_list=[3,10], temperature=0.1, check_memory=False):
    torch.cuda.empty_cache()
    
    if check_memory: print_gpu_memory()  # Before anything
    
    model.train()
    if check_memory: print_gpu_memory()  # After model.train()
    
    optimizer.zero_grad()
    if check_memory: print_gpu_memory()   # After zero_grad
    
    # Flatten positions
    pos_flat = pos.view(-1, 3)
    batch_idx = torch.arange(pos.size(0), device=pos.device).repeat_interleave(pos.size(1))
    if check_memory: print_gpu_memory()   # After flattening
    
    # Build graphs with memory checks
    edge_indices = []
    edge_attrs = []
    for k in k_list:
        with torch.no_grad():  # Disable grad for graph building
            
            if check_memory: print_gpu_memory() 
            edge_index, edge_attr = model._build_pbc_knn(pos_flat, batch_idx, k)
            if check_memory: print_gpu_memory() # After graph build
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)
    
    # Forward passes
    h0 = model.encoder(pos_flat)
    if check_memory: print_gpu_memory()   # After encoder
    
    projections = []
    for edge_index, edge_attr in zip(edge_indices, edge_attrs):
        h = h0
        for layer in model.layers:
            h = layer(h, edge_index, edge_attr)
        projections.append(model.projection_head(h))
        if check_memory: print_gpu_memory()   # After each projection
    
    # Loss calculation
    loss1 = nt_xent_loss(projections[0], projections[1], temperature)
    loss2 = nt_xent_loss(projections[1], projections[0], temperature)
    loss = (loss1 + loss2) / 2
    if check_memory: print_gpu_memory()   # After loss
    
    # Backward
    loss.backward()
    if check_memory: print_gpu_memory()  # After backward
    
    optimizer.step()
    if check_memory: print_gpu_memory()   # After step
    with torch.no_grad():
        acc1 = nt_xent_accuracy(projections[0], projections[1], temperature)
        acc2 = nt_xent_accuracy(projections[1], projections[0], temperature)
        acc = (acc1 + acc2) / 2
    
    return loss.item(), acc

def train_step(model, pos, optimizer, k_list, temperature):
    """Memory-optimized training step"""
    torch.cuda.empty_cache()
    
    # Manual garbage collection
    import gc
    gc.collect()
    
    model.train()
    optimizer.zero_grad(set_to_none=True)  # More memory efficient
    
    # Process in no_grad for graph building
    with torch.no_grad():
        pos_flat = pos.view(-1, 3)
        batch_idx = torch.arange(pos.size(0), device=pos.device).repeat_interleave(pos.size(1))
        
        # Build graphs separately to free memory
        edge_indices = []
        edge_attrs = []
        for k in k_list:
            edge_index, edge_attr = model._build_pbc_knn(pos_flat, batch_idx, k)
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)
            del edge_index, edge_attr
    
    h0 = model.encoder(pos_flat)
    projections = []
    
    for edge_index, edge_attr in zip(edge_indices, edge_attrs):
        h = h0.clone()  # Prevent memory sharing
        for layer in model.layers:
            h = layer(h, edge_index, edge_attr)
        proj = model.projection_head(h)
        projections.append(proj)
        del h, proj
    
    loss1 = nt_xent_loss(projections[0], projections[1], temperature)
    loss2 = nt_xent_loss(projections[1], projections[0], temperature)
    loss = (loss1 + loss2) / 2
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    with torch.no_grad():
        acc1 = nt_xent_accuracy(projections[0], projections[1], temperature)
        acc2 = nt_xent_accuracy(projections[1], projections[0], temperature)
        acc = (acc1 + acc2) / 2
    
    return loss.item(), acc

def evaluate(model, dataloader, k_list=[5, 30], temperature=0.1):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    batch_count = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                pos = batch[0]
            else:
                pos = batch
                
            pos = pos.to(device)
            batch_size, num_particles, _ = pos.shape
            
            # Prepare batched input
            pos_flat = pos.view(-1, 3)
            batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_particles)
            
            # Generate views
            projections = model(pos_flat, batch=batch_idx, k=k_list, mode='contrastive')
            z1, z2 = projections
            
            # Calculate loss
            loss1 = nt_xent_loss(z1, z2, temperature)
            loss2 = nt_xent_loss(z2, z1, temperature)
            loss = (loss1 + loss2) / 2
            total_loss += loss.item()
            batch_count += 1

            acc1 = nt_xent_accuracy(z1, z2, temperature)
            acc2 = nt_xent_accuracy(z2, z1, temperature)
            acc = (acc1 + acc2) / 2
            total_metric += acc
            
    return total_loss / batch_count, total_metric / batch_count

def train(model, train_loader, val_loader, k_list=[5, 30], lr=0.001, temperature=0.1, 
          num_epochs=100, save_path=None, patience=10, min_lr=1e-6, factor=0.5, 
          grad_clip=1.0, use_amp=True):
    """Enhanced training loop with learning rate scheduling and early stopping"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', 
        factor=factor, 
        patience=patience//2,
        min_lr=min_lr
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == 'cuda')
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    lr_history = []

    for epoch in tqdm(range(num_epochs), desc='Training Epochs'):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        batch_count = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            if isinstance(batch, (list, tuple)):
                pos = batch[0]
            else:
                pos = batch
                
            pos = pos.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
                # Prepare batched input
                pos_flat = pos.view(-1, 3)
                batch_idx = torch.arange(pos.size(0), device=device).repeat_interleave(pos.size(1))
                
                # Generate views
                projections = model(pos_flat, batch=batch_idx, k=k_list, mode='contrastive')
                z1, z2 = projections
                
                # Calculate loss
                loss1 = nt_xent_loss(z1, z2, temperature)
                loss2 = nt_xent_loss(z2, z1, temperature)
                loss = (loss1 + loss2) / 2
                
                # Calculate accuracy
                acc1 = nt_xent_accuracy(z1, z2, temperature)
                acc2 = nt_xent_accuracy(z2, z1, temperature)
                acc = (acc1 + acc2) / 2
            
            # Backpropagation with gradient clipping
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_loss += loss.item()
            epoch_train_acc += acc
            batch_count += 1
        
        # Calculate epoch metrics
        avg_train_loss = epoch_train_loss / batch_count
        avg_train_acc = epoch_train_acc / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        
        # Validation phase
        avg_val_loss, avg_val_acc = evaluate(model, val_loader, k_list, temperature)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        scheduler.step(avg_val_loss)
        
        # Print metrics
        tqdm.write(f'Epoch {epoch+1}/{num_epochs} | '
              f'LR: {current_lr:.2e} | '
              f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | '
              f'Train Acc: {avg_train_acc:.2%} | Val Acc: {avg_val_acc:.2%}')
        
        # Checkpointing and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                tqdm.write(f"Saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                tqdm.write(f'Early stopping after {patience} epochs without improvement')
                break
                
    # Load best model weights at the end
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        tqdm.write(f"Loaded best model (val_loss={best_val_loss:.4f})")
    
    return model, train_losses, val_losses, train_accs, val_accs, lr_history