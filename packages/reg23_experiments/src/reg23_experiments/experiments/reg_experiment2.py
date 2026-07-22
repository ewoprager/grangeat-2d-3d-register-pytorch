import copy
import logging
import pprint
from typing import Any

import matplotlib

matplotlib.use("QtAgg")

import pandas as pd
import torch
import traitlets
from jaxtyping import Float64

from reg23_experiments.data.structs import Cropping, Error, Transformation
from reg23_experiments.experiments.helpers import ParametrisedSimilarityMetric, string_to_sim_met
from reg23_experiments.experiments.registration import RegConfig, run_reg
from reg23_experiments.ops import geometry
from reg23_experiments.ops.data_manager import args_from_dadg, data_manager
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation, \
    mapping_transformation_to_parameters, random_parameters_at_distance
from reg23_experiments.utils.console_logging import tqdm
from reg23_experiments.experiments.batched import objective_function_binary_weighted

__all__ = ["ExperimentConfig", "run_experiment", "exp_config_from_dict"]

logger = logging.getLogger(__name__)


class ExperimentConfig(traitlets.HasTraits):
    ct_path: str = traitlets.Unicode(default_value=traitlets.Undefined)
    xray_path: str = traitlets.Unicode(default_value=traitlets.Undefined)
    downsample_level: int = traitlets.Int(min=0, default_value=traitlets.Undefined)
    # truncation_percent: int = traitlets.Int(min=0, max=100, default_value=traitlets.Undefined)
    desired_h_valid: int = traitlets.Float(min=1.0, max=100.0, default_value=traitlets.Undefined)
    crop_min_size: float = traitlets.Float(min=0.0, default_value=traitlets.Undefined)
    weight_alpha: float = traitlets.Float(min=0.0, default_value=traitlets.Undefined)
    sim_metric: str = traitlets.Enum(values=[  #
        "zncc",  #
        "local_zncc",  #
    ], default_value=traitlets.Undefined)
    starting_distance: float = traitlets.Float(default_value=traitlets.Undefined)
    sample_count_per_distance: int = traitlets.Int(min=1, default_value=traitlets.Undefined)
    reg_config: RegConfig = traitlets.Instance(RegConfig, allow_none=False, default_value=traitlets.Undefined)


def run_experiment(  #
        exp_config: ExperimentConfig,  #
        device: torch.device,  #
        tqdm_position: int = 0,  #
        dry_run: bool = False,  #
        batch_size: int = 1,  #
) -> pd.DataFrame | None:
    """
    Run multiple (`sample_count_per_distance`) registrations according to the given parameters, and return the average
    distance from ground truth at each iteration.
    :param exp_config:
    :param device:
    :param tqdm_position:
    :return: A tensor of size (iteration count,) or None; the distance from g.t. of the optimisation at each
    iteration, averaged over `sample_count_per_distance` repetitions, unless the configuration is trivial /
    unnecessary, in which case `None`.
    """
    data_manager().set("ct_path", exp_config.ct_path, check_equality=True)
    data_manager().set("xray_path", exp_config.xray_path, check_equality=True)
    data_manager().set("downsample_level", exp_config.downsample_level, check_equality=True)
    # data_manager().set("truncation_percent", exp_config.truncation_percent, check_equality=True)
    data_manager().set("desired_h_valid", exp_config.desired_h_valid)
    # -----
    # Configuring according to desired similarity metric
    p_sim_met: ParametrisedSimilarityMetric = string_to_sim_met(exp_config.sim_metric)

    # -----
    # Defining the objective function
    def objective_function(parameters: Float64[torch.Tensor, "6"]) -> torch.Tensor:
        return args_from_dadg(  #
            names_left=["weighted_sim_metric", "parameters"]  #
        )(objective_function_binary_weighted)(  #
            weighted_sim_metric=p_sim_met.func_weighted,  #
            parameters=parameters.unsqueeze(0),  #
        )[0]

    def objective_function_batched(parameters: Float64[torch.Tensor, "b 6"]) -> Float64[torch.Tensor, "b"]:
        return args_from_dadg(  #
            names_left=["weighted_sim_metric", "parameters"]  #
        )(objective_function_binary_weighted)(  #
            weighted_sim_metric=p_sim_met.func_weighted,  #
            parameters=parameters,  #
        )

    # -----
    # Running repeated registrations with configured parameters
    dimensionality = 6
    distance_samples = torch.empty(
        [int(exp_config.sample_count_per_distance), int(exp_config.reg_config.iteration_count)], dtype=torch.float64,
        device=device)  # size = (sample count, iteration count)
    transformation_gt: Transformation | None | Error = data_manager().get("transformation_gt")
    if isinstance(transformation_gt, Error):
        raise Exception(f"Failed to get ground truth transformation: {transformation_gt.description}")
    if transformation_gt is None:
        raise Exception(f"No ground truth transformation available.")
    ground_truth = mapping_transformation_to_parameters(transformation_gt)
    for i in tqdm(  #
            range(int(exp_config.sample_count_per_distance)),  #
            desc="Repeated samples",  #
            position=tqdm_position,  #
            leave=None  #
    ):
        starting_params = random_parameters_at_distance(ground_truth, exp_config.starting_distance)
        # -----
        # Crop to the non-zero domain of the DRR at the starting parameters
        data_manager().set("current_transformation", mapping_parameters_to_transformation(starting_params))
        cropping: Cropping = args_from_dadg()(geometry.get_crop_nonzero_drr)()
        if cropping.is_collapsed(exp_config.crop_min_size):
            cropping = cropping.uncollapse(exp_config.crop_min_size)
        data_manager().set("further_cropping", cropping, check_equality=True)
        # -----
        # Registration
        if not dry_run:
            res = run_reg(  #
                obj_fun=objective_function if batch_size == 1 else objective_function_batched,  #
                config=exp_config.reg_config,  #
                starting_params=starting_params,  #
                device=device,  #
                tqdm_position=tqdm_position + 1,  #
                batch_size=batch_size,  #
            )  # size = (iteration count, dimensionality + 1)
            distance_samples[i, :] = torch.linalg.vector_norm(res[:, 0:dimensionality] - ground_truth,
                                                              dim=1)  # size = (iteration count,)

    return None if dry_run else pd.DataFrame({  #
        "iteration": torch.arange(exp_config.reg_config.iteration_count).numpy(),  # size = (iteration count,)
        "distance": distance_samples.mean(dim=0).cpu().numpy(),  # size = (iteration count,)
        "distance_std": distance_samples.std(dim=0).cpu().numpy(),  #
    })


def exp_config_from_dict(dict_config: dict[str, Any]) -> ExperimentConfig | Error:
    dict_config_copy = copy.deepcopy(dict_config)
    try:
        reg_config = RegConfig(  #
            particle_count=dict_config_copy.pop("particle_count"),  #
            particle_initialisation_spread=dict_config_copy.pop("particle_initialisation_spread"),  #
            iteration_count=dict_config_copy.pop("iteration_count")  #
        )
    except Exception as e:
        return Error(f"Failed to construct RegConfig: {e}\nParameters:\n{pprint.pformat(dict_config_copy)}")

    dict_config_copy["reg_config"] = reg_config

    try:
        exp_config = ExperimentConfig(**dict_config_copy)
    except Exception as e:
        return Error(f"Failed to construct ExperimentConfig: {e}\nParameters:\n{pprint.pformat(dict_config_copy)}")

    return exp_config
