from typing import NamedTuple, Callable
import copy

import numpy as np
import pyswarms
import torch

from program import data_manager, args_from_dag
from registration.lib.structs import Transformation, GrowingTensor
from registration.lib.optimisation import mapping_transformation_to_parameters, mapping_parameters_to_transformation

__all__ = ["OptimisationResult", "pso"]


class OptimisationResult(NamedTuple):
    params: torch.Tensor
    param_history: torch.Tensor
    value_history: torch.Tensor


@args_from_dag(names_left=["transformation"])
def pso(*, transformation: Transformation, objective_function: Callable[[torch.Tensor], torch.Tensor],
        pso_particle_count: int, pso_iteration_count: int,
        iteration_callback: Callable[[torch.Tensor, torch.Tensor], None] | None = None) -> OptimisationResult:
    starting_parameters = mapping_transformation_to_parameters(transformation)
    n_dimensions = starting_parameters.numel()
    param_history = GrowingTensor([n_dimensions], pso_particle_count * pso_iteration_count)
    value_history = GrowingTensor([], pso_particle_count * pso_iteration_count)

    param_history.push_back(starting_parameters)
    value_history.push_back(objective_function(starting_parameters))

    def objective_pso(particle_params: np.ndarray) -> np.ndarray:
        ret = np.zeros(particle_params.shape[0])
        for i, row in enumerate(particle_params):
            params = torch.tensor(copy.deepcopy(row))
            param_history.push_back(params)
            value = objective_function(params)
            value_history.push_back(value)
            iteration_callback(param_history.get(), value_history.get())
            ret[i] = value.item()
        return ret

    options = {'c1': 2.525, 'c2': 1.225, 'w': 0.28}  # {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    initial_positions = np.random.normal(loc=starting_parameters.numpy(), size=(pso_particle_count, n_dimensions),
                                         scale=1.0)
    initial_positions[0] = starting_parameters.numpy()
    optimiser = pyswarms.single.GlobalBestPSO(n_particles=pso_particle_count, dimensions=n_dimensions,
                                              init_pos=initial_positions, options=options)

    cost, converged_params = optimiser.optimize(objective_pso, iters=pso_iteration_count)
    return OptimisationResult(params=torch.from_numpy(converged_params), param_history=param_history.get(),
                              value_history=value_history.get())
