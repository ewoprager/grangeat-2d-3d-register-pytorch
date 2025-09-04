from typing import Callable
import logging

logger = logging.getLogger(__name__)

import torch


def local_search(*, starting_position: torch.Tensor, initial_step_size: torch.Tensor, objective_function: Callable[[
    torch.Tensor], torch.Tensor], step_size_reduction_ratio: float = .75, no_improvement_threshold: int = 10,
                 max_iterations: int = 5000, max_reductions: int = 4) -> torch.Tensor:
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
