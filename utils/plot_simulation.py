import torch
import plotly.graph_objs as go

def plot_simulation_summary(simulation_summary):
    # Extract values from tuples and move to CPU
    thetas = [theta for theta, phi, rss in simulation_summary]
    phis = [phi for theta, phi, rss in simulation_summary]
    rss_vals = [rss for theta, phi, rss in simulation_summary]

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=thetas,
        y=phis,
        z=rss_vals,
        mode='markers+lines',
        marker=dict(
            size=6,
            color=rss_vals,     # Color by RSS value
            colorscale='Viridis',
            colorbar=dict(title='RSS (dB)')
        ),
        line=dict(
            color='gray',
            width=2
        )
    )])

    # Axis labels and layout
    fig.update_layout(
        title='Beamforming Simulation Search Path',
        scene=dict(
            xaxis_title='Theta (°)',
            yaxis_title='Phi (°)',
            zaxis_title='RSS (dB)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show(renderer = "iframe")