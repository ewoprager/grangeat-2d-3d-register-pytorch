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

__all__ = ["ExperimentConfig", "run_experiment", "exp_config_from_dict"]

logger = logging.getLogger(__name__)


class ExperimentConfig(traitlets.HasTraits):
    ct_path: str = traitlets.Unicode(default_value=traitlets.Undefined)
    xray_path: str = traitlets.Unicode(default_value=traitlets.Undefined)
    downsample_level: int = traitlets.Int(min=0, default_value=traitlets.Undefined)
    # truncation_percent: int = traitlets.Int(min=0, max=100, default_value=traitlets.Undefined)
    desired_h_valid: int = traitlets.Float(min=1.0, max=100.0, default_value=traitlets.Undefined)
    cropping: str = traitlets.Enum(values=[  #
        "None",  #
        "nonzero_drr",  #
        "full_depth_drr"  #
    ], default_value=traitlets.Undefined)
    crop_min_size: float = traitlets.Float(min=0.0, default_value=traitlets.Undefined)
    crop_expand: float = traitlets.Float(default_value=traitlets.Undefined)
    mask: str = traitlets.Enum(values=[  #
        "None",  #
        "Every evaluation",  #
        "Every evaluation weighting zncc"  #
    ], default_value=traitlets.Undefined)
    sim_metric: str = traitlets.Enum(values=[  #
        "zncc",  #
        "local_zncc",  #
        "multiscale_zncc",  #
        "gradient_correlation"  #
    ], default_value=traitlets.Undefined)
    starting_distance: float = traitlets.Float(default_value=traitlets.Undefined)
    sample_count_per_distance: int = traitlets.Int(min=1, default_value=traitlets.Undefined)
    reg_config: RegConfig = traitlets.Instance(RegConfig, allow_none=False, default_value=traitlets.Undefined)


def run_experiment(  #
        exp_config: ExperimentConfig,  #
        device: torch.device,  #
        tqdm_position: int = 0,  #
        dry_run: bool = False,  #
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
    # Configuring according to desired masking technique
    if exp_config.mask == "None":
        apply_mask = False
        data_manager().set("mask_transformation", None, check_equality=True)
    elif exp_config.mask == "Every evaluation":
        apply_mask = True
        weight_with_mask = False
    elif exp_config.mask == "Every evaluation weighting zncc":
        apply_mask = True
        weight_with_mask = True
        # Checking that the parametrised sim. metric has a weighted counterpart
        if p_sim_met.func_weighted is None:
            # No weighted counterpart of the similarity metric; skipping this configuration
            return None
    else:
        raise ValueError(f"Unknown mask technique '{exp_config.mask}'.")

    # -----
    # Defining the objective function
    def objective_function(parameters: Float64[torch.Tensor, "6"]) -> torch.Tensor:
        t: Transformation = mapping_parameters_to_transformation(parameters)
        # Setting the parameters
        data_manager().set("current_transformation", t)
        if apply_mask:
            data_manager().set("mask_transformation", t)
        # Getting the resulting moving and fixed images
        moving_image: torch.Tensor | Error = data_manager().get("moving_image")
        fixed_image: torch.Tensor | Error = data_manager().get("fixed_image")
        # Comparing, potentially weighting with a mask
        if apply_mask and weight_with_mask:
            mask: torch.Tensor | Error = data_manager().get("mask")
            return -p_sim_met.func_weighted(moving_image, fixed_image, mask)
        return -p_sim_met.func(moving_image, fixed_image)

    # -----
    # Running repeated registrations with configured parameters
    dimensionality = 6
    distance_samples = torch.empty(
        [int(exp_config.sample_count_per_distance), int(exp_config.reg_config.iteration_count)], dtype=torch.float64,
        device=device)  # size = (sample count, iteration count)
    crop_size_samples = torch.empty([int(exp_config.sample_count_per_distance), 2],
                                    dtype=torch.float64)  # size = (sample count, 2); (width, height)
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
        # Configuring according to desired cropping technique
        data_manager().set("current_transformation", mapping_parameters_to_transformation(starting_params))
        if exp_config.cropping == "None":
            cropping: Cropping | None = None
        elif exp_config.cropping == "nonzero_drr":
            cropping: Cropping | None = args_from_dadg()(geometry.get_crop_nonzero_drr)()
        elif exp_config.cropping == "full_depth_drr":
            cropping: Cropping | None = args_from_dadg()(geometry.get_crop_full_depth_drr)()
        else:
            raise ValueError(f"Unknown cropping technique '{exp_config.cropping}'.")
        if not dry_run:
            image: torch.Tensor | Error = data_manager().get("image_2d_full")
            if isinstance(image, Error):
                raise Exception(f"Failed to get image_2d_full: {image.description}")
            spacing: torch.Tensor | Error = data_manager().get("image_2d_full_spacing")
            if isinstance(spacing, Error):
                raise Exception(f"Failed to get image_2d_full_spacing: {spacing.description}")
            spacing = spacing.cpu()
            if cropping is None:
                crop_size_samples[i, 0] = float(image.size()[1]) * spacing[0].item()
                crop_size_samples[i, 1] = float(image.size()[0]) * spacing[1].item()
            else:
                if cropping.is_collapsed(exp_config.crop_min_size):
                    cropping = cropping.uncollapse(exp_config.crop_min_size)
                crop_size_samples[i, 0] = (cropping.right - cropping.left) * float(image.size()[1]) * spacing[0].item()
                crop_size_samples[i, 1] = (cropping.bottom - cropping.top) * float(image.size()[0]) * spacing[1].item()
                cropping = cropping.expand_mm(exp_config.crop_expand, image_size=image.size(), image_spacing=spacing)
                # expand could be negative, so checking again for collapse
                if cropping.is_collapsed(exp_config.crop_min_size):
                    cropping = cropping.uncollapse(exp_config.crop_min_size)

        data_manager().set("further_cropping", cropping, check_equality=True)
        # -----
        # Registration
        if not dry_run:
            res = run_reg(  #
                obj_fun=objective_function,  #
                config=exp_config.reg_config,  #
                starting_params=starting_params,  #
                device=device,  #
                tqdm_position=tqdm_position + 1)  # size = (iteration count, dimensionality + 1)
            distance_samples[i, :] = torch.linalg.vector_norm(res[:, 0:dimensionality] - ground_truth,
                                                              dim=1)  # size = (iteration count,)

    return None if dry_run else pd.DataFrame({  #
        "iteration": torch.arange(exp_config.reg_config.iteration_count).numpy(),  # size = (iteration count,)
        "distance": distance_samples.mean(dim=0).cpu().numpy(),  # size = (iteration count,)
        "distance_std": distance_samples.std(dim=0).cpu().numpy(),  #
        "crop_width": crop_size_samples[:, 0].mean().cpu().numpy(),  #
        "crop_height": crop_size_samples[:, 1].mean().cpu().numpy(),  #
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
