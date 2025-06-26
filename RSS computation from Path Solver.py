from sionna.rt import *
import torch
import math
import numpy as np


path_solver = PathSolver()
def compute_rss_from_path_solver(theta, phi):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = path_solver(scene, max_depth=5, los=True, specular_reflection=True, diffuse_reflection=False, refraction=True)
    
    # --- 1. Extract CIR from paths ---
    cir = paths.cir(out_type='torch')[0].to(torch.complex64).to(device)

    # --- 2. Compute Precoding Vector ---
    w = compute_precoding_vector_torch(num_rows, num_cols, 0.5, 0.5, theta, phi).T  # Shape: [Rows * Columns, N]
    w = w.to(torch.complex64).to(device)

    # --- 3. Channel Matrix (sum over paths) ---
    H = cir.sum(dim=-1).sum(dim=-1)
    H = H.reshape(-1, H.shape[2] * H.shape[3])

    # --- 4. Beamforming: y = H @ w ---
    y = torch.matmul(H, w)  # Shape: [Rx, N]

    # --- 5. RSS = sum |y|^2 across Rx antennas ---
    rss = torch.sum(torch.abs(y)**2, dim=0)  # Shape: [N]

    # --- 6. Convert to dB ---
    rss_db = 10 * torch.log10(rss) +13.71

    return rss