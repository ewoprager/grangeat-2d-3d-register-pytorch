import torch

from reg23_experiments.data.structs import OptimisationInstance
from reg23_experiments.ops.swarm import Swarm, SwarmConfig

__all__ = ["PsoInstance"]


class PsoInstance(OptimisationInstance):
    def __init__(self, *, particle_count: int, starting_pos: torch.Tensor, starting_spread: float,
                 config: SwarmConfig, device: torch.device):
        self._swarm = Swarm(config=config, dimensionality=6,
                            particle_count=particle_count, initialisation_position=starting_pos,
                            initialisation_spread=torch.full_like(starting_pos, starting_spread), device=device)

    def name(self) -> str:
        return "PSO"

    def step(self) -> bool:
        self._swarm.iterate()
        return False

    def get_best_position(self) -> torch.Tensor:
        return self._swarm.current_optimum_position

    def get_best(self) -> torch.Tensor:
        return self._swarm.current_optimum
