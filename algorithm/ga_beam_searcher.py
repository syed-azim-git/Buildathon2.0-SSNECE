import math
import torch
from typing import Callable, Optional, Tuple
import random


class GABeamSearcherTorch:
    def __init__(
        self,
        path_solver: any,
        scene: any,
        fitness_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        population_size: int = 30,
        max_generations: int = 50,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        angle_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 90), (0, 360)),
        seed_particle: Optional[Tuple[float, float]] = None,
        resolution: float = 5.0,
        search_limit: float = 30.0,
        stagnant_limit: int = 4,
        elite_count: int = 3,
        angle_unit: str = "degree",
        min_std_threshold: float = 0.05,
        num_cols: int = 1,
        num_rows: int = 1
    ):
        self.fitness_fn = fitness_fn
        self.pop_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.angle_bounds = angle_bounds
        self.seed_particle = seed_particle
        self.resolution = resolution
        self.search_limit = search_limit
        self.stagnant_limit = stagnant_limit
        self.elite_count = elite_count
        self.angle_unit = angle_unit.lower()
        self.min_std_threshold = min_std_threshold
        self.angle_history = []
        self.convergence_rss = []
        self.path_solver = path_solver
        self.scene = scene
        self.num_cols = num_cols
        self.num_rows = num_rows

        self.factor = math.pi / 180 if self.angle_unit == "degree" else 1.0
        self.theta_bounds = tuple(f * self.factor for f in self.angle_bounds[0])
        self.phi_bounds = tuple(f * self.factor for f in self.angle_bounds[1])
        self.search_limit *= self.factor
        if self.seed_particle:
            self.seed_particle = (seed_particle[0] * self.factor, seed_particle[1] * self.factor)

        self.elite_buffer: List[Tuple[float, float]] = []

    def _sample_population(self, center_theta, center_phi, device):
        theta_offset = (torch.rand(self.pop_size, device=device) * 2 - 1) * self.search_limit
        phi_offset = (torch.rand(self.pop_size, device=device) * 2 - 1) * self.search_limit

        theta = center_theta + theta_offset
        phi = center_phi + phi_offset

        theta = torch.remainder(theta, math.pi)
        phi = torch.remainder(phi, 2 * math.pi)

        resolution = self.resolution * self.factor
        theta = torch.round(theta / resolution) * resolution
        phi = torch.round(phi / resolution) * resolution

        return theta, phi

    def _crossover(self, t1, p1, t2, p2):
        mask = torch.rand_like(t1) < 0.5
        return torch.where(mask, t1, t2), torch.where(mask, p1, p2)

    def _mutate(self, theta, phi):
        noise_t = (torch.rand_like(theta) - 0.5) * 2 * self.resolution * self.factor
        noise_p = (torch.rand_like(phi) - 0.5) * 2 * self.resolution * self.factor
        theta = torch.remainder(theta + noise_t, math.pi)
        phi = torch.remainder(phi + noise_p, 2 * math.pi)
        return theta, phi

    def search(self) -> Tuple[float, float, float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.seed_particle:
            theta, phi = self._sample_population(self.seed_particle[0], self.seed_particle[1], device)
        else:
            theta = torch.rand(self.pop_size, device=device) * math.pi
            phi = torch.rand(self.pop_size, device=device) * 2 * math.pi

        self.angle_history.extend([
            (math.degrees(t.item()), math.degrees(p.item())) for t, p in zip(theta, phi)
        ])

        fitness = self.fitness_fn(theta, phi, self.path_solver, self.scene, self.num_rows, self.num_cols)
        best_idx = torch.argmax(fitness)
        best_theta = theta[best_idx].clone()
        best_phi = phi[best_idx].clone()
        best_rss = fitness[best_idx].clone()
        stagnant_counter = 0

        for gen in range(self.max_generations):
            self.elite_buffer.append((best_theta.item(), best_phi.item()))
            if len(self.elite_buffer) > self.elite_count:
                self.elite_buffer.pop(0)

            theta, phi = self._sample_population(best_theta, best_phi, device)
            fitness = self.fitness_fn(theta, phi, self.path_solver, self.scene, self.num_rows, self.num_cols)

            print(f"[GEN {gen:03}] RSS Max: {fitness.max().item():.2f} | Std: {fitness.std().item():.4f}")

            if fitness.std().item() < self.min_std_threshold:
                print(f"[STOP] RSS standard deviation below threshold: {fitness.std().item():.4f}")
                break

            top_k = torch.topk(fitness, k=self.pop_size // 2).indices
            parents_theta = theta[top_k]
            parents_phi = phi[top_k]

            elite_theta = torch.tensor([e[0] for e in self.elite_buffer], device=device)
            elite_phi = torch.tensor([e[1] for e in self.elite_buffer], device=device)

            combined_theta = torch.cat([parents_theta, elite_theta])
            combined_phi = torch.cat([parents_phi, elite_phi])

            new_theta, new_phi = [], []
            for _ in range(self.pop_size):
                i1, i2 = random.sample(range(len(combined_theta)), 2)
                t1, p1 = combined_theta[i1], combined_phi[i1]
                t2, p2 = combined_theta[i2], combined_phi[i2]

                if random.random() < self.crossover_rate:
                    child_theta, child_phi = self._crossover(t1, p1, t2, p2)
                else:
                    child_theta, child_phi = t1.clone(), p1.clone()

                if random.random() < self.mutation_rate or True:  # Always some noise
                    child_theta, child_phi = self._mutate(child_theta.unsqueeze(0), child_phi.unsqueeze(0))
                    child_theta, child_phi = child_theta[0], child_phi[0]

                new_theta.append(child_theta)
                new_phi.append(child_phi)

            theta = torch.stack(new_theta)
            phi = torch.stack(new_phi)
            fitness = self.fitness_fn(theta, phi)
            gen_best_idx = torch.argmax(fitness)

            if fitness[gen_best_idx] > best_rss:
                best_theta = theta[gen_best_idx].clone()
                best_phi = phi[gen_best_idx].clone()
                best_rss = fitness[gen_best_idx].clone()
                self.seed_particle = (best_theta.item(), best_phi.item())
                stagnant_counter = 0
            else:
                stagnant_counter += 1
                if stagnant_counter >= self.stagnant_limit:
                    print(f"[STOP] No improvement for {self.stagnant_limit} generations.")
                    break
                    
        self.convergence_rss.append(best_rss.item())
        resolution = self.resolution * self.factor
        best_theta = torch.round(best_theta / resolution) * resolution
        best_phi = torch.round(best_phi / resolution) * resolution

        if self.angle_unit == "degree":
            return (
                torch.rad2deg(best_theta),
                torch.rad2deg(best_phi),
                best_rss
            )
        else:
            return best_theta, best_phi, best_rss


    def plot_convergence(self):
        if not hasattr(self, "convergence_rss"):
            print("No convergence data found.")
            return
    
        plt.figure(figsize=(8, 4))
        plt.plot(self.convergence_rss, marker="o")
        plt.xlabel("Generation")
        plt.ylabel("Best RSS")
        plt.title("Genetic Algorithm Convergence")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_angle_heatmap(angle_history, resolution=1.0):
        df = pd.DataFrame(angle_history, columns=["theta_deg", "phi_deg"])
        df["theta_bin"] = (df["theta_deg"] / resolution).round() * resolution
        df["phi_bin"] = (df["phi_deg"] / resolution).round() * resolution
        heatmap = df.groupby(["theta_bin", "phi_bin"]).size().reset_index(name="count")
        fig = px.density_heatmap(
            heatmap,
            x="phi_bin",
            y="theta_bin",
            z="count",
            nbinsx=int(360 / resolution),
            nbinsy=int(90 / resolution),
            color_continuous_scale="Turbo",
            labels={"phi_bin": "Azimuth (°)", "theta_bin": "Elevation (°)", "count": "Hits"},
            title="Angle Exploration Heatmap"
        )
        fig.update_yaxes(autorange="reversed")
        fig.show()
