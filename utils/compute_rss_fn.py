
from sionna.rt import *
from utils.compute_precoding_vector import compute_precoding_vector_torch
import mitsuba as mi
import torch
import numpy as np


def compute_rss(theta_deg, phi_deg):
    precoding_vec = compute_precoding_vector_torch(num_rows, num_cols, v_spacing, h_spacing, theta_deg, phi_deg)
    precoding_vec_cpu = precoding_vec.cpu().numpy()

    precoding_vec_real = mi.TensorXf(precoding_vec_cpu.real.astype(np.float32))
    precoding_vec_imag = mi.TensorXf(precoding_vec_cpu.imag.astype(np.float32))

    precoding_vec = zip(precoding_vec_real,precoding_vec_imag)
    rss = []
    for i in precoding_vec:
        coverage_map = rm_solver(
            scene=scene,
            cell_size=[1, 1],
            samples_per_tx=int(1e8),
            max_depth=7,
            precoding_vec=i,
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=False
        )
    
        rss_tensor = coverage_map.rss.numpy()
        cell_centers = coverage_map.cell_centers.numpy()
        _, num_cells_y, num_cells_x = rss_tensor.shape
    
        flat_centers = cell_centers.reshape(-1, 3)
        rx_pos = np.array(rx.position).reshape(1, 3)
        distances = np.linalg.norm(flat_centers - rx_pos, axis=1)
        nearest_index = np.argmin(distances)
    
        y_idx = nearest_index // num_cells_x
        x_idx = nearest_index % num_cells_x
        rss_linear = rss_tensor[0, y_idx, x_idx]
        #rss.append(torch.tensor(rss_linear))
        rss.append(10 * torch.log10(torch.tensor(rss_linear)) if rss_linear > 0 else torch.tensor(-200))
    return torch.stack(rss)
