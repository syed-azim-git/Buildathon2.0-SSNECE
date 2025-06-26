import matplotlib.pyplot as plt
import io
from PIL import Image
import tqdm
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

rss_start = compute_rss_from_path_solver(torch.tensor([0]),torch.tensor([0])).item()
# paths = path_solver(scene, max_depth = 5)

simulation_summary = []

# simulation_summary.append((0, 0, rss_start, h))
simulation_summary.append((0, 0, rss_start))


# Prepare to store frames
frames = []
buf = io.BytesIO()
fig.savefig(buf, format='png')
buf.seek(0)
frames.append(Image.open(buf))
plt.close(fig)
rm_solver = RadioMapSolver()
# Define receiver movement parameters
num_frames = 42
x_positions = np.linspace(-30, 180, num_frames)

for x in tqdm.tqdm(x_positions):
    # Update receiver position
    rx.position = np.array([-70, x, 1.5])

    theta, phi, rss = pso.search()
    print(rss)

    precoding_vec = compute_precoding_vector_torch(
        rows = num_rows,
        cols= num_cols,
        v_space = v_spacing,
        h_space = h_spacing, 
        theta_deg = theta.reshape(-1), 
        phi_deg = phi.reshape(-1)
    ).cpu().numpy()

    coverage_map = rm_solver(
        scene=scene,
        cell_size=[0.5, 0.5],
        samples_per_tx=int(1e7),
        max_depth=7,
        precoding_vec=(
            mi.TensorXf(precoding_vec.real.astype(np.float32)),
            mi.TensorXf(precoding_vec.imag.astype(np.float32))
        ),
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        refraction=False
    )

    # h = path_solver(scene, max_depth=5).cir()[0]  # Just CIR tensor


    # simulation_summary.append((
    #     theta.item(),
    #     phi.item(),
    #     rss.item(),
    #     h
    # ))

    simulation_summary.append((
        theta.item(),
        phi.item(),
        rss.item()
    ))
    
    # Render scene with coverage map
    fig = scene.render(
        camera=camera,
        radio_map=coverage_map,
        rm_metric="rss",
        rm_db_scale=True
    )

    buf = io.BytesIO()

    # Save the figure to a BytesIO object
    fig.savefig(buf, format='png')
    buf.seek(0)
    frames.append(Image.open(buf))
    plt.close(fig)

frames[0].save(
    'coverage_animation.gif',
    save_all=True,
    append_images=frames[1:],
    duration=200,  # duration between frames in milliseconds
    loop=0  # loop indefinitely
)
