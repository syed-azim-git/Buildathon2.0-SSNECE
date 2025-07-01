import torch
import math


def compute_precoding_vector_torch(rows, cols, v_space, h_space, theta_deg, phi_deg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    theta = torch.deg2rad(theta_deg)  # shape: [N]
    phi = torch.deg2rad(phi_deg)      # shape: [N]

    theta = theta.to(device)
    phi = phi.to(device)

    row_indices = torch.arange(rows, dtype=torch.float32, device=device)  # [R]
    col_indices = torch.arange(cols, dtype=torch.float32, device=device)  # [C]

    # [N, R], [N, C]
    phase_row = 2 * math.pi * v_space * torch.sin(theta[:, None]) * row_indices[None, :]
    phase_col = 2 * math.pi * h_space * torch.sin(phi[:, None]) * col_indices[None, :]

    # [N, R], [N, C]
    steer_row = torch.exp(1j * phase_row)
    steer_col = torch.exp(1j * phase_col)

    # [N, R, C]
    steering_matrix = steer_row[:, :, None] * steer_col[:, None, :]

    # [N, R*C]
    precoding_vec = steering_matrix.reshape(steering_matrix.shape[0], -1)

    # Normalize each vector (across dim 1)
    precoding_vec = precoding_vec / torch.linalg.norm(precoding_vec, dim=1, keepdim=True)

    return precoding_vec  # shape: [N, R*C]