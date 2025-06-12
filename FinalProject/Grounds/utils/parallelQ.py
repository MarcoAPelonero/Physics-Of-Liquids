import torch
import math
from tqdm import tqdm 
import numpy as np

max_factorial = 40
factorials = torch.ones(max_factorial + 1, dtype=torch.float64)
for i in range(1, max_factorial + 1):
    factorials[i] = factorials[i - 1] * i

def compute_normalization_constants(max_l):
    """Precompute spherical harmonics normalization constants for l=0 to max_l"""
    norm_consts = torch.zeros((max_l + 1, max_l + 1), dtype=torch.float64)
    for l_val in range(max_l + 1):
        for m_val in range(0, l_val + 1):
            num = factorials[l_val - m_val]
            den = factorials[l_val + m_val]
            norm_val = math.sqrt((2 * l_val + 1) * num / (4 * math.pi * den))
            norm_consts[l_val, m_val] = norm_val
    return norm_consts

def compute_associated_legendre(l, x):
    """Compute associated Legendre polynomials P_l^m(x) for m=0 to l"""
    n_points = x.shape[0]
    P = torch.zeros((l + 1, l + 1, n_points), dtype=x.dtype, device=x.device)
    
    P[0, 0] = 1.0
    if l >= 1:
        P[1, 0] = x
        P[1, 1] = -torch.sqrt(1.0 - x**2)
    
    for ell in range(2, l + 1):
        for m in range(0, ell):
            term1 = (2 * ell - 1) * x * P[ell-1, m]
            term2 = (ell - 1 + m) * P[ell-2, m]
            P[ell, m] = (term1 - term2) / (ell - m)
        P[ell, ell] = (2 * ell - 1) * torch.sqrt(1.0 - x**2) * P[ell-1, ell-1]
    
    return P

def compute_spherical_harmonics(l, theta, phi, norm_consts):
    """
    Return a tensor of shape (n_points, 2*l+1) with columns ordered
    [m = -l, …, -1, 0, 1, …, +l].
    """
    x = torch.clamp(torch.cos(theta), -1.0, 1.0)

    P = compute_associated_legendre(l, x)         

    Y_pos = []
    for m in range(l + 1):
        norm = norm_consts[l, m].to(x)             
        Y_pos.append(norm * P[l, m] * torch.exp(1j * m * phi))

    Y_neg = [((-1.0) if m % 2 else 1.0) * torch.conj(Y_pos[m])
             for m in range(1, l + 1)]             
    Y_neg.reverse()                                

    Y_all = torch.stack(Y_neg + Y_pos, dim=1)      
    return Y_all


def compute_Ql(positions, l_list, r_cutoff, box=None):
    """
    positions: (N,3) tensor of coordinates
    l_list:   list of integers (e.g., [4,6])
    r_cutoff: float, cutoff radius
    box:      (3,2) periodic boundaries [[xmin,xmax], ...]
    """
    device = positions.device
    dtype = positions.dtype
    pos = positions
    N = pos.shape[0]
    
    max_l = max(l_list) if l_list else 0
    norm_consts = compute_normalization_constants(max_l)
    
    disp = pos.unsqueeze(1) - pos.unsqueeze(0)
    if box is not None:
        lengths = (box[:, 1] - box[:, 0]).view(1, 3)
        for d in range(3):
            L = lengths[0, d]
            disp[:, :, d] = disp[:, :, d] - L * torch.round(disp[:, :, d] / L)
    
    dist = torch.norm(disp, dim=-1)
    neighbor_mask = (dist > 0) & (dist < r_cutoff)
    mask_idx = neighbor_mask.nonzero(as_tuple=False)
    i_idx, j_idx = mask_idx[:, 0], mask_idx[:, 1]
    unique, counts = torch.unique(i_idx, return_counts=True)
    print(f"\nNeighbor counts summary:")
    print(f"Min neighbors: {counts.min().item()}")
    print(f"Max neighbors: {counts.max().item()}")
    print(f"Avg neighbors: {counts.float().mean().item():.1f}")
    rij = disp[i_idx, j_idx]
    r = dist[i_idx, j_idx]
    theta = torch.acos(torch.clamp(rij[:, 2] / r, -1.0, 1.0))  
    phi = torch.atan2(rij[:, 1], rij[:, 0])                    

    results = {}
    for ell in tqdm(l_list, desc="Computing coefficients", leave=True):
        Y_all = compute_spherical_harmonics(ell, theta, phi, norm_consts)
        
        q_lm = torch.zeros((N, 2*ell+1), dtype=torch.cfloat, device=device)
        q_lm.index_add_(0, i_idx, Y_all)
        
        nei_counts = neighbor_mask.sum(dim=1).clamp(min=1)
        q_lm = q_lm / nei_counts.view(-1, 1).to(q_lm.dtype)
        
        q_l_atom = torch.sqrt((4 * torch.pi / (2*ell+1)) * (q_lm.abs()**2).sum(dim=1))
        q_lm_mean = q_lm.mean(dim=0)
        Q_l_global = torch.sqrt((4 * torch.pi / (2*ell+1)) * (q_lm_mean.abs()**2).sum()).item()
        Q_l = q_l_atom.mean().item()  
        
        results[ell] = {
            'q_l_atom': q_l_atom,
            'Q_l': Q_l,
            'Q_l_global': Q_l_global,
        }
    
    return results

def compute_Ql_from_atoms(atoms, l_list, r_cutoff, box=None):
    """
    atoms:     list of atom dicts with 'x','y','z'
    l_list:    list of integers (e.g., [4,6])
    r_cutoff:  float, cutoff radius
    box:       (3,2) periodic boundaries
    """
    positions = torch.tensor(
        [[atom['x'], atom['y'], atom['z']] for atom in atoms],
        dtype=torch.float32
    )
    return compute_Ql(positions, l_list, r_cutoff, box)

def compute_rdf(positions, box, r_max=None, dr=0.01):
    """
    positions : (N,3) real-space coordinates
    box       : array_like shape (3,2) of [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
    r_max     : float, maximum distance (default = half the smallest box length)
    dr        : bin width
    Returns  : (r_centers, g_r)
    """
    N = positions.shape[0]
    L = np.array([box[i][1] - box[i][0] for i in range(3)])
    V = L.prod()
    rho = N / V

    if r_max is None:
        r_max = 0.5 * L.min()

    edges = np.arange(0.0, r_max + dr, dr)
    shell_vol = 4.0/3.0 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    r_centers = 0.5*(edges[1:]+edges[:-1])

    dists = []
    for i in range(N-1):
        delta = positions[i+1:] - positions[i]
        delta -= L * np.round(delta / L)
        dist_ij = np.linalg.norm(delta, axis=1)
        dists.append(dist_ij)
    dists = np.concatenate(dists)

    counts, _ = np.histogram(dists, bins=edges)
    norm = rho * shell_vol * N  
    g_r = counts / norm

    return r_centers, g_r