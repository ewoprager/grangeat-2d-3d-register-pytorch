from typing import Callable, Literal

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import torch
import traitlets

from reg23_experiments.ops import swarm as pso
from reg23_experiments.ops.data_manager import data_manager
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation
from reg23_experiments.utils.console_logging import tqdm

__all__ = ["RegConfig", "run_reg"]


class RegConfig(traitlets.HasTraits):
    particle_count: int = traitlets.Int(default_value=traitlets.Undefined)
    particle_initialisation_spread: float = traitlets.Float(default_value=traitlets.Undefined)
    iteration_count: int = traitlets.Int(default_value=traitlets.Undefined)


def run_reg(  #
        *,  #
        obj_fun: Callable,  #
        starting_params: torch.Tensor,  #
        config: RegConfig,  #
        device: torch.device,  #
        plot: Literal["no", "yes", "mask"] = "no",  #
        tqdm_position: int = 0,  #
        batch_size: int = 1,  #
) -> torch.Tensor:
    """
    Run a PSO from the given starting params and return a tensor containing the params and O.F. value at each iteration.
    :param obj_fun:
    :param starting_params:
    :param config:
    :param device:
    :param plot:
    :param tqdm_position:
    :return: A tensor of size (iteration count, dimensionality + 1), where each row corresponds to an iteration of
    the optimisation, and stores the following data: | <- position of current best -> | current best |
    """
    if plot != "no":
        ncols = 2
        if plot == "mask":
            ncols += 2
        fig, axes = plt.subplots(1, ncols)
        axes = axes.tolist()
        # axes.insert(2, axes[1].twinx())
        plt.ion()
        plt.show()
        t = mapping_parameters_to_transformation(starting_params)
        axes[0].clear()
        axes[0].set_title("moving image AT start: R=({:.3f},{:.3f},{:.3f}), T=({:.3f},{:.3f},{:.3f})".format(  #
            t.rotation[0].item(), t.rotation[1].item(), t.rotation[2].item(), t.translation[0].item(),
            t.translation[1].item(), t.translation[2].item()))
        data_manager().set("current_transformation", mapping_parameters_to_transformation(starting_params))
        axes[0].imshow(data_manager().get("moving_image").cpu().numpy())
        plt.draw()
        plt.pause(0.1)

    # -----
    # Initialise a particle swarm optimisation, with tqdm
    pso_config = pso.SwarmConfig(objective_function=obj_fun)
    dimensionality = starting_params.numel()
    # initialise the return tensor
    ret = torch.empty([config.iteration_count, dimensionality + 1], dtype=torch.float32, device=device)
    tqdm_iterator = tqdm(range(config.iteration_count), desc="PSO iterations", position=tqdm_position, leave=None)
    # initialise the swarm, which performs an o.f. evaluation for each particle
    swarm = pso.Swarm(  #
        config=pso_config,  #
        dimensionality=dimensionality,  #
        particle_count=config.particle_count,  #
        initialisation_position=starting_params,  #
        initialisation_spread=torch.full([dimensionality], config.particle_initialisation_spread),  #
        device=device,  #
        batch_size=batch_size,  #
    )
    ret[0, 0:dimensionality] = swarm.current_optimum_position.to(dtype=torch.float32, device=device)
    ret[0, -1] = swarm.current_optimum.to(dtype=torch.float32, device=device)
    tqdm_iterator.update()
    # -----
    # The optimisation loop
    for it in range(1, config.iteration_count):
        swarm.iterate()
        ret[it, 0:dimensionality] = swarm.current_optimum_position.to(dtype=torch.float32, device=device)
        ret[it, -1] = swarm.current_optimum.to(dtype=torch.float32, device=device)

        if plot != "no":
            data_manager().set("current_transformation",
                               mapping_parameters_to_transformation(swarm.current_optimum_position))
            axes[0].clear()
            axes[0].imshow(data_manager().get("moving_image").cpu().numpy())
            t = data_manager().get("current_transformation")
            axes[0].set_title("Iteration {}: R=({:.3f},{:.3f},{:.3f}), T=({:.3f},{:.3f},{:.3f})".format(  #
                it, t.rotation[0].item(), t.rotation[1].item(), t.rotation[2].item(), t.translation[0].item(),
                t.translation[1].item(), t.translation[2].item()))
            axes[1].clear()
            axes[1].plot(ret[0:it + 1, -1].cpu().numpy())
            axes[1].set_xlabel("iteration")
            axes[1].set_ylabel("o.f. value")
            if plot == "mask":
                axes[2].clear()
                axes[2].set_title("mask")
                axes[2].imshow(data_manager().get("mask").cpu().numpy())
                axes[3].clear()
                axes[3].set_title("masked fixed image")
                axes[3].imshow(data_manager().get("fixed_image").cpu().numpy())
            plt.draw()
            plt.pause(0.1)

        tqdm_iterator.update()
        tqdm_iterator.set_postfix(best=swarm.current_optimum.item())
    return ret
