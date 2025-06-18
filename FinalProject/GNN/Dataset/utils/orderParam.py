import numpy as np
import math

def compute_normalization_constants(max_l):
    """Precompute spherical harmonics normalization constants for l=0 to max_l"""
    max_fact = 2 * max_l
    factorials = np.zeros(max_fact + 1, dtype=np.float64)
    factorials[0] = 1.0
    for i in range(1, max_fact + 1):
        factorials[i] = factorials[i - 1] * i

    norm_consts = np.zeros((max_l + 1, max_l + 1), dtype=np.float64)
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
    P = np.zeros((l + 1, l + 1, n_points), dtype=x.dtype)
    
    P[0, 0] = 1.0
    if l >= 1:
        P[1, 0] = x
        P[1, 1] = -np.sqrt(1.0 - x**2)
    
    for ell in range(2, l + 1):
        for m in range(0, ell):
            term1 = (2 * ell - 1) * x * P[ell-1, m]
            term2 = (ell - 1 + m) * P[ell-2, m]
            P[ell, m] = (term1 - term2) / (ell - m)
        P[ell, ell] = (2 * ell - 1) * np.sqrt(1.0 - x**2) * P[ell-1, ell-1]
    
    return P

def compute_spherical_harmonics(l, theta, phi, norm_consts):
    """
    Return a tensor of shape (n_points, 2*l+1) with columns ordered
    [m = -l, …, -1, 0, 1, …, +l].
    """
    x = np.clip(np.cos(theta), -1.0, 1.0)
    P = compute_associated_legendre(l, x)
    
    Y_pos = []
    for m in range(l + 1):
        norm = norm_consts[l, m]
        Y_pos.append(norm * P[l, m] * np.exp(1j * m * phi))
    
    Y_neg = [((-1)**m) * np.conj(Y_pos[m]) for m in range(1, l + 1)]
    Y_neg.reverse()
    
    Y_all = np.stack(Y_neg + Y_pos, axis=1)
    return Y_all

def compute_Ql_knn(positions, l_list, n_neighbors, box=None):
    """
    positions : (N,3) array of coordinates
    l_list    : list[int] – e.g. [4, 6]
    n_neighbors: int      – number of nearest neighbours to use for every atom
    box       : (3,2) periodic boundaries [[xmin,xmax], [ymin,ymax], [zmin,zmax]]
    """
    if n_neighbors < 1:
        raise ValueError("`n_neighbors` must be ≥ 1")
    N = positions.shape[0]
    if n_neighbors >= N:
        raise ValueError("`n_neighbors` must be smaller than the number of atoms")

    max_l = max(l_list) if l_list else 0
    norm_consts = compute_normalization_constants(max_l)

    disp = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N, N, 3)
    
    if box is not None:
        lengths = (box[:, 1] - box[:, 0]).reshape(1, 3)
        for d in range(3):
            L = lengths[0, d]
            disp[:, :, d] -= L * np.round(disp[:, :, d] / L)

    dist = np.linalg.norm(disp, axis=-1)  # (N, N)
    np.fill_diagonal(dist, np.finfo(dist.dtype).max)  

    knn_idx = np.argsort(dist, axis=1)[:, :n_neighbors]  
    i_idx = np.repeat(np.arange(N), n_neighbors)
    j_idx = knn_idx.ravel()

    rij = disp[i_idx, j_idx] 
    r = np.linalg.norm(rij, axis=1)
    r_safe = np.where(r > 1e-10, r, 1.0) 
    theta = np.arccos(np.clip(rij[:, 2] / r_safe, -1.0, 1.0))
    phi = np.arctan2(rij[:, 1], rij[:, 0])

    results = {}
    for ell in l_list:
        Y_all = compute_spherical_harmonics(ell, theta, phi, norm_consts) 
        
        q_lm = np.zeros((N, 2 * ell + 1), dtype=np.complex128)
        np.add.at(q_lm, i_idx, Y_all)
        q_lm /= n_neighbors  
        q_l_atom = np.sqrt((4 * np.pi / (2 * ell + 1)) * 
                          np.linalg.norm(q_lm, axis=1))
        
        q_lm_mean = np.mean(q_lm, axis=0)
        Q_l_global = np.sqrt((4 * np.pi / (2 * ell + 1)) *
                            np.sum(np.abs(q_lm_mean)**2))
        Q_l = np.mean(q_l_atom)

        results[ell] = {
            "atom_indices": [
                {"index": i, "q_l_atom": q_l_atom[i].item()}
                for i in range(N)
            ],
            "Q_l": Q_l,
            "Q_l_global": Q_l_global
        }

    return results