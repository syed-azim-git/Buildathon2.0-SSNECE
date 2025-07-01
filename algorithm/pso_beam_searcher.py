import math
import torch
from typing import Callable, Optional, Tuple


class PSOBeamSearcherTorch:
    def __init__(
        self,
        path_solver: any,
        scene: any,
        fitness_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        population_size: int = 30,
        max_generations: int = 50,
        w: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5,
        search_resolution: float = 5.0,
        search_window: float = 30.0,
        stagnant_limit: int = 4,
        seed_particle: Optional[Tuple[float, float]] = None,
        angle_unit: str = "degree",
        num_cols: int = 1,
        num_rows: int = 1
    ):
        self.fitness_fn = fitness_fn
        self.pop_size = population_size
        self.max_gen = max_generations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.search_resolution = search_resolution
        self.search_window = search_window
        self.stagnant_limit = stagnant_limit
        self.angle_unit = angle_unit.lower()
        self.seed_particle = seed_particle
        self.path_solver = path_solver
        self.scene = scene
        self.num_cols = num_cols
        self.num_rows = num_rows

        self.factor = math.pi / 180 if self.angle_unit == "degree" else 1.0
        self.res = self.search_resolution * self.factor
        self.window = self.search_window * self.factor

    def _initialize_particles(self, device):
        # Generate candidates in ±30° window around seed
        if self.seed_particle is None:
            raise ValueError("seed_particle must be provided for local search.")

        seed_theta, seed_phi = self.seed_particle
        seed_theta *= self.factor
        seed_phi *= self.factor

        theta_range = torch.arange(
            seed_theta - self.window,
            seed_theta + self.window + self.res,
            self.res, device=device
        ).clamp(-math.pi / 2, math.pi / 2)  # theta ∈ [0, π/2] ##

        phi_range = torch.arange(
            seed_phi - self.window,
            seed_phi + self.window + self.res,
            self.res, device=device
        ) % (2 * math.pi)  # phi ∈ [0, 2π]

        theta = theta_range[torch.randint(0, len(theta_range), (self.pop_size,))]
        phi = phi_range[torch.randint(0, len(phi_range), (self.pop_size,))]

        return theta, phi

    def _round_to_res(self, angles: torch.Tensor) -> torch.Tensor:
        return torch.round(angles / self.res) * self.res

    def search(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.seed_particle)
        theta, phi = self._initialize_particles(device)
        v_theta = torch.zeros_like(theta)
        v_phi = torch.zeros_like(phi)

        fitness = self.fitness_fn(theta, phi,  self.path_solver, self.scene, self.num_rows, self.num_cols).to(device)
        pbest_theta, pbest_phi, pbest_rss = theta.clone(), phi.clone(), fitness.clone()

        best_idx = torch.argmax(fitness)
        gbest_theta = pbest_theta[best_idx].clone()
        gbest_phi = pbest_phi[best_idx].clone()
        gbest_rss = pbest_rss[best_idx].clone()

        stagnant_counter = 0

        for _ in range(self.max_gen):
            r1, r2 = torch.rand(self.pop_size, device=device), torch.rand(self.pop_size, device=device)

            v_theta = self.w * v_theta + self.c1 * r1 * (pbest_theta - theta) + self.c2 * r2 * (gbest_theta - theta)
            v_phi = self.w * v_phi + self.c1 * r1 * (pbest_phi - phi) + self.c2 * r2 * (gbest_phi - phi)

            theta = torch.clamp(theta + v_theta, 0, math.pi / 2)
            phi = (phi + v_phi) % (2 * math.pi)

            theta = self._round_to_res(theta)
            phi = self._round_to_res(phi)

            fitness = self.fitness_fn(theta, phi,  self.path_solver, self.scene, self.num_rows, self.num_cols).to(device)

            better = fitness > pbest_rss
            pbest_theta = torch.where(better, theta, pbest_theta)
            pbest_phi = torch.where(better, phi, pbest_phi)
            pbest_rss = torch.where(better, fitness, pbest_rss)

            best_idx = torch.argmax(pbest_rss)
            if pbest_rss[best_idx] > gbest_rss:
                gbest_theta = pbest_theta[best_idx].clone()
                gbest_phi = pbest_phi[best_idx].clone()
                gbest_rss = pbest_rss[best_idx].clone()
                self.seed_particle = (gbest_theta.item(), gbest_phi.item())
                stagnant_counter = 0
            else:
                stagnant_counter += 1
                if stagnant_counter >= self.stagnant_limit and gbest_rss.item() > -58:
                    break

        if self.angle_unit == "degree":
            return torch.rad2deg(gbest_theta), torch.rad2deg(gbest_phi), gbest_rss
        else:
            return gbest_theta, gbest_phi, gbest_rss