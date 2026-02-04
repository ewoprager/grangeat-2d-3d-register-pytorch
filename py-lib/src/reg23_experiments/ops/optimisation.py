from typing import Callable, NamedTuple
import logging
import copy

import torch
import numpy as np
import pyswarms

from reg23_experiments.data.structs import Transformation, GrowingTensor
from reg23_experiments.ops.data_manager import args_from_dag

__all__ = ["mapping_transformation_to_parameters", "mapping_parameters_to_transformation", "local_search",
           "random_parameters_at_distance"]

logger = logging.getLogger(__name__)


def mapping_transformation_to_parameters(transformation: Transformation) -> torch.Tensor:
    ret = transformation.vectorised()
    return ret * torch.tensor([32.0, 32.0, 32.0, 1.0, 1.0, 0.05], device=ret.device, dtype=ret.dtype)


def mapping_parameters_to_transformation(params: torch.Tensor) -> Transformation:
    return Transformation.from_vector(
        params * torch.tensor([0.03125, 0.03125, 0.03125, 1.0, 1.0, 20.0], device=params.device, dtype=params.dtype))


def random_parameters_at_distance(from_parameters: torch.Tensor, distance: float | torch.Tensor) -> torch.Tensor:
    u_hat = torch.nn.functional.normalize(torch.randn_like(from_parameters), dim=0)
    return from_parameters + distance * u_hat


def local_search(*, starting_position: torch.Tensor, initial_step_size: torch.Tensor,
                 objective_function: Callable[[torch.Tensor], torch.Tensor], step_size_reduction_ratio: float = .75,
                 no_improvement_threshold: int = 10, max_iterations: int = 5000,
                 max_reductions: int = 4) -> torch.Tensor:
    assert (starting_position.size() == initial_step_size.size())

    x = starting_position
    f = objective_function(x)
    reductions = 0
    no_improvement_count = 0
    step_size = initial_step_size
    for _ in range(max_iterations):
        new = torch.normal(x, step_size)
        new_f = objective_function(new)
        if new_f < f:
            x = new
            f = new_f
        else:
            no_improvement_count += 1
            if no_improvement_count >= no_improvement_threshold:
                if reductions >= max_reductions:
                    break
                step_size *= step_size_reduction_ratio
                reductions += 1
                no_improvement_count = 0

    return x


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
