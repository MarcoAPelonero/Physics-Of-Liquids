import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def nt_xent_loss_2n_3d(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    Symmetric (2 × N-row) NT-Xent loss for 3D inputs [B, N, dim] with chunk processing.
    """
    B, N, dim = z1.shape
    device = z1.device
    z = torch.cat([z1, z2], dim=0)  # [2B, N, dim]
    z = F.normalize(z, dim=-1)
    z_flat = z.reshape(-1, dim)  # [2B*N, dim]
    total_samples = 2 * B * N
    total_loss = torch.tensor(0.0, device=device)
    
    for b in range(2 * B):
        anchors = z[b]  # [N, dim]
        base_idx = b * N  # Starting index for the current batch's anchors
        # Determine positive indices: same position in the other augmented view
        if b < B:
            pos_indices = base_idx + B * N + torch.arange(N, device=device)
        else:
            pos_indices = base_idx - B * N + torch.arange(N, device=device)
        
        pos_logits = torch.einsum('nd,nd->n', anchors, z_flat[pos_indices]) / temperature  # [N]
        
        log_denom = torch.full((N,), -float('inf'), device=device)
        
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            chunk = z_flat[start:end]  # [chunk_size, dim]
            logits_chunk = torch.einsum('nd,md->nm', anchors, chunk) / temperature  # [N, chunk_size]
            
            diag_mask = (base_idx + torch.arange(N, device=device)[:, None] == torch.arange(start, end, device=device))
            logits_chunk[diag_mask] = -float('inf')
            
            max_vals = logits_chunk.max(dim=1, keepdim=True)[0]
            exp_logits = torch.exp(logits_chunk - max_vals)
            log_sum_exp = max_vals.squeeze(1) + torch.log(exp_logits.sum(dim=1))
            log_denom = torch.logaddexp(log_denom, log_sum_exp)
        
        batch_loss = - (pos_logits - log_denom)  # [N]
        total_loss += batch_loss.sum()
    
    return total_loss / (2 * B * N)

def nt_xent_accuracy_2n_3d(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    chunk_size: int = 512,
) -> float:
    """
    Top-1 contrastive accuracy for 3D inputs [B, N, dim] with chunk processing.
    """
    B, N, dim = z1.shape
    device = z1.device
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)  # [2B, N, dim]
    z_flat = z.reshape(-1, dim)  # [2B*N, dim]
    total_samples = 2 * B * N
    correct = 0
    
    for b in range(2 * B):
        anchors = z[b]  # [N, dim]
        base_idx = b * N  # Starting index for the current batch's anchors
        
        if b < B:
            pos_indices = base_idx + B * N + torch.arange(N, device=device)
        else:
            pos_indices = base_idx - B * N + torch.arange(N, device=device)
        
        max_logits = torch.full((N,), -float('inf'), device=device)
        pred_indices = torch.zeros((N,), dtype=torch.long, device=device)
        
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            chunk = z_flat[start:end] 
            logits_chunk = torch.einsum('nd,md->nm', anchors, chunk) / temperature  
            
            diag_mask = (base_idx + torch.arange(N, device=device)[:, None] == torch.arange(start, end, device=device))
            logits_chunk[diag_mask] = -float('inf')
            
            chunk_max, chunk_argmax = logits_chunk.max(dim=1)
            update_mask = chunk_max > max_logits
            if update_mask.any():
                chunk_global_argmax = start + chunk_argmax
                max_logits[update_mask] = chunk_max[update_mask]
                pred_indices[update_mask] = chunk_global_argmax[update_mask]
        
        correct += (pred_indices == pos_indices).sum().item()
    
    return correct / (2 * B * N)

# Test the 3D version
if __name__ == "__main__":
    z1 = torch.randn(3, 10, 128)
    z2 = torch.randn(3, 10, 128)
    
    loss = nt_xent_loss_2n_3d(z1, z2)
    print(f"3D NT-Xent Loss: {loss.item()}")
    
    accuracy = nt_xent_accuracy_2n_3d(z1, z2)
    print(f"3D NT-Xent Accuracy: {accuracy:.4f}")