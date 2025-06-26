precoding_vec = compute_precoding_vector_torch(num_rows, num_cols, v_spacing, h_spacing, torch.tensor([0]), torch.tensor([0])).cpu().numpy()

rm_solver=RadioMapSolver()
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

fig = scene.render(
        camera=camera,
        radio_map=coverage_map,
        rm_metric="rss",
        rm_db_scale=False
    )