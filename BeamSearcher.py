searcher = GABeamSearcherTorch(
    fitness_fn=compute_rss_from_path_solver,
    population_size = 50,
    max_generations = 100,
    mutation_rate = 0.7,
    crossover_rate = 0.5,
    seed_particle=(0, 0),
    search_limit=40,
    angle_bounds=((0, 90), (0, 360)),
    resolution=0.5,
    elite_count=3,#
    stagnant_limit = 20,
    angle_unit='degree'
)

rss_start = compute_rss_from_path_solver(torch.tensor([0]),torch.tensor([0])).item()
