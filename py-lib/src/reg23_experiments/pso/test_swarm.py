import logging

import torch
import matplotlib.pyplot as plt

from reg23_experiments.pso.swarm import OptimisationConfig, Swarm

__all__ = ["test_swarm"]

logger = logging.getLogger(__name__)


def test_swarm():
    particle_count = 20

    def objective_function(xy: torch.Tensor) -> torch.Tensor:
        p = xy[0] - 3.0
        q = xy[1] - 2.0
        return p * p + q * q

    config = OptimisationConfig(objective_function=objective_function, inertia_coefficient=0.9,
                                cognitive_coefficient=0.5, social_coefficient=0.3)
    swarm = Swarm(config=config, dimensionality=2, particle_count=particle_count,
                  initialisation_position=torch.tensor([0.0, 0.0]), initialisation_spread=torch.tensor([10.0, 10.0]),
                  device=torch.device("cpu"))
    iteration_count = 50
    global_bests = torch.zeros([iteration_count])
    for i in range(iteration_count):
        swarm.iterate()
        global_bests[i] = swarm.current_optimum

    if False:
        plt.plot(global_bests)
        plt.show()
