import copy
import logging
import pprint
from typing import Any, Literal

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import traitlets
from jaxtyping import Float64

from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.experiments.batched import objective_function_alpha_weighted, \
    objective_function_binary_weighted, objective_function_together
from reg23_experiments.experiments.dadg_updaters import batched
from reg23_experiments.experiments.registration import RegConfig, run_reg
from reg23_experiments.ops.data_manager import args_from_dadg, dadg_updater, data_manager
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation, \
    mapping_transformation_to_parameters
from reg23_experiments.utils.console_logging import indentation_prefix, tqdm

__all__ = ["ExperimentConfig", "run_experiment", "exp_config_from_dict"]

logger = logging.getLogger(__name__)


class ExperimentConfig(traitlets.HasTraits):
    """
    Notes:
        - the value of `weighting` is interpreted as follows:
            - if None, no weight image is used
            - if "linear", the scaling image is applied as a weight image
            - if of type float, the weighting image is generated from the scaling image, using the float value as alpha
        - the value of `iterations_per_update` is interpreted as follows:
            - if 0, the cropping, scaling and weight images will be updated to the current transformation every o.f. 
            evaluation
            - otherwise, these images will only be updated every N iterations
    """

    # ----- images
    ct_path: str = traitlets.Unicode(default_value=traitlets.Undefined)
    xray_path: str = traitlets.Unicode(default_value=traitlets.Undefined)
    # ----- preprocessing
    downsample_level: int = traitlets.Int(min=0, default_value=traitlets.Undefined)
    truncation_percent: int = traitlets.Int(min=0, max=100, default_value=traitlets.Undefined)
    # ----- cropping
    cropping_method: Literal["none", "bounding_box", "valid_only"] = traitlets.Enum(values=[  #
        "none",  #
        "bounding_box",  #
        "valid_only",  #
    ], default_value=traitlets.Undefined)
    crop_min_size: float = traitlets.Float(min=0.0, default_value=traitlets.Undefined)
    iterations_per_crop_update: int = traitlets.Int(min=0,
                                                    default_value=traitlets.Undefined)  # 0 means every o.f. eval.
    # ----- scaling
    apply_scaling: bool = traitlets.Bool(default_value=traitlets.Undefined)
    # ----- similarity & weighting
    weighting: None | Literal["linear"] | float = traitlets.Union(trait_types=[  #
        traitlets.Enum(values=["linear"], default_value=traitlets.Undefined),  #
        traitlets.Float(min=0.0, default_value=traitlets.Undefined),  #
    ], allow_none=True, default_value=traitlets.Undefined)
    iterations_per_weight_update: int = traitlets.Int(min=0,
                                                      default_value=traitlets.Undefined)  # 0 means every o.f. eval.
    sim_metric: Literal["zncc", "local_zncc"] = traitlets.Enum(values=[  #
        "zncc",  #
        "local_zncc",  #
    ], default_value=traitlets.Undefined)
    # ----- registration
    starting_distance: float = traitlets.Float(default_value=traitlets.Undefined)
    sample_count_per_distance: int = traitlets.Int(min=1, default_value=traitlets.Undefined)
    reg_config: RegConfig = traitlets.Instance(RegConfig, allow_none=False, default_value=traitlets.Undefined)


def run_experiment(  #
        config: ExperimentConfig,  #
        device: torch.device,  #
        tqdm_position: int = 0,  #
        dry_run: bool = False,  #
        batch_size: int = 1,  #
        plot: bool = False,  #
) -> pd.DataFrame | None:
    """
    Run multiple (`sample_count_per_distance`) registrations according to the given parameters, and return the average
    distance from ground truth at each iteration.
    :param config:
    :param device:
    :param tqdm_position:
    :return: A tensor of size (iteration count,) or None; the distance from g.t. of the optimisation at each
    iteration, averaged over `sample_count_per_distance` repetitions, unless the configuration is trivial /
    unnecessary, in which case `None`.
    """
    data_manager().set("ct_path", config.ct_path, check_equality=True)
    data_manager().set("xray_path", config.xray_path, check_equality=True)
    data_manager().set("downsample_level", config.downsample_level, check_equality=True)
    data_manager().set("truncation_percent", config.truncation_percent, check_equality=True)
    # -----
    # Configuring according to desired similarity metric
    # p_sim_met: ParametrisedSimilarityMetric = string_to_sim_met(config.sim_metric)
    data_manager().set("cropping_method", config.cropping_method, check_equality=True)
    data_manager().set("crop_min_size", config.crop_min_size, check_equality=True)
    data_manager().set("apply_scaling", config.apply_scaling, check_equality=True)
    data_manager().set("weighting", config.weighting, check_equality=True)
    data_manager().set("sim_metric", config.sim_metric, check_equality=True)

    # -----
    # Defining the objective function
    def objective_function(parameters: Float64[torch.Tensor, "6"]) -> torch.Tensor:
        if config.weight_alpha < 1.0e-4:
            return args_from_dadg(  #
                names_left=["weighted_sim_metric", "parameters"]  #
            )(objective_function_binary_weighted)(  #
                weighted_sim_metric=p_sim_met.func_weighted,  #
                parameters=parameters.unsqueeze(0),  #
            )[0]
        else:
            return args_from_dadg(  #
                names_left=["weighted_sim_metric", "parameters", "weight_alpha"]  #
            )(objective_function_alpha_weighted)(  #
                weighted_sim_metric=p_sim_met.func_weighted,  #
                parameters=parameters.unsqueeze(0),  #
                weight_alpha=config.weight_alpha,  #
            )[0]

    def objective_function_batched(parameters: Float64[torch.Tensor, "b 6"]) -> Float64[torch.Tensor, "b"]:
        if config.weight_alpha < 1.0e-4:
            return args_from_dadg(  #
                names_left=["weighted_sim_metric", "parameters"]  #
            )(objective_function_binary_weighted)(  #
                weighted_sim_metric=p_sim_met.func_weighted,  #
                parameters=parameters,  #
            )
        else:
            return args_from_dadg(  #
                names_left=["weighted_sim_metric", "parameters", "weight_alpha"]  #
            )(objective_function_alpha_weighted)(  #
                weighted_sim_metric=p_sim_met.func_weighted,  #
                parameters=parameters,  #
                weight_alpha=config.weight_alpha,  #
            )

    def new_objective_function(parameters: Float64[torch.Tensor, "b 6"]) -> Float64[torch.Tensor, "b"]:
        err: Error | None = data_manager().set("parameters", parameters)
        if isinstance(err, Error):
            logger.warning(f"Error setting parameters in o.f.: {err.description}")
        res: torch.Tensor | Error = data_manager().get("of_values")
        if isinstance(res, Error):
            raise Exception(f"Objective function evaluation failed: {res}")
        return res

    def new_new_objective_function(parameters: Float64[torch.Tensor, "b 6"]) -> Float64[torch.Tensor, "b"]:
        return args_from_dadg(  #
            names_left=["parameters"]  #
        )(objective_function_together)(  #
            parameters=parameters,  #
        )

    # -----
    # Periodic behaviour
    periodic_behaviour = []
    # Crop rectangle updates
    if config.iterations_per_crop_update > 0:
        data_manager().remove_updater("refresh_cropping")

        def do_crop_refresh(best_parameters: Float64[torch.Tensor, "6"]) -> None:
            # get the cropping at the best transformation
            value_dict: dict[str, Any] | Error = args_from_dadg(names_left=["parameters"])(batched.refresh_cropping)(
                parameters=best_parameters.unsqueeze(0))
            if isinstance(value_dict, Error):
                raise Exception(f"Error refreshing cropping for crop refresh: {value_dict.description}")
            best_cropping = value_dict["further_cropping"]
            # set the cropping
            data_manager().set("further_cropping", best_cropping)

        periodic_behaviour.append((config.iterations_per_crop_update, do_crop_refresh))
    else:
        batch_size = 1  # necessary for crop refresh every o.f. evaluation
        data_manager().add_updater(  #
            "refresh_cropping",  #
            dadg_updater(names_returned=["further_cropping"])(batched.refresh_cropping),  #
        )
    # Weight image updates, only after the crop updates
    if config.iterations_per_weight_update > 0:
        data_manager().remove_updater("refresh_weights")

        def do_weight_refresh(best_parameters: Float64[torch.Tensor, "6"]) -> None:
            # get the scaling image at the best transformation
            if isinstance(err := data_manager().set("parameters", best_parameters.unsqueeze(0)), Error):
                raise Exception(f"Error setting parameters for weight refresh: {err.description}")
            value_dict: dict[str, Any] | Error = args_from_dadg()(batched.refresh_scaling_images)()
            if isinstance(value_dict, Error):
                raise Exception(f"Error refreshing scaling image for weight refresh: {value_dict.description}")
            best_scaling_image = value_dict["scaling_images"]
            # convert to weight image
            value_dict: dict[str, Any] | Error = args_from_dadg(names_left=["scaling_images"])(batched.refresh_weights)(
                scaling_images=best_scaling_image)
            if isinstance(value_dict, Error):
                raise Exception(f"Error refreshing weight image for weight refresh: {value_dict.description}")
            weight_image = value_dict["weight_images"]
            # set the weight image
            data_manager().set("weight_images", weight_image)

        periodic_behaviour.append((config.iterations_per_weight_update, do_weight_refresh))
    else:
        data_manager().add_updater(  #
            "refresh_weights",  #
            dadg_updater(names_returned=["weight_images"])(batched.refresh_weights),  #
        )

    if plot:
        # Show the target image, fixed image and moving image at the gold-standard
        fig, axes = plt.subplots(1, 3)

        # Target image
        image_2d_full: torch.Tensor | Error = data_manager().get("image_2d_full")
        if isinstance(image_2d_full, Error):
            raise RuntimeError(f"Error getting image_2d_full: {image_2d_full.description}")
        axes[0].imshow(image_2d_full.cpu().numpy())
        axes[0].set_title("original target")
        # Initialise at the gold-standard
        transformation_gt: Transformation | None | Error = data_manager().get("transformation_gt")
        if isinstance(transformation_gt, Error):
            raise Exception(f"Failed to get ground truth transformation: {transformation_gt.description}")
        if transformation_gt is None:
            raise Exception(f"No ground truth transformation available.")
        params_gt = mapping_transformation_to_parameters(transformation_gt)
        if isinstance(err := data_manager().set("parameters", params_gt.unsqueeze(0)), Error):
            raise RuntimeError(f"Error setting parameters to ground truth transformation: {err.description}")
        for _, f in periodic_behaviour:
            f(params_gt)
        # Fixed image at gold-standard
        fixed_image: torch.Tensor | Error = data_manager().get("fixed_images")
        if isinstance(fixed_image, Error):
            raise RuntimeError(f"Error getting fixed image: {fixed_image.description}")
        axes[1].imshow(fixed_image[0].cpu().numpy())
        axes[1].set_title("fixed image")
        # Moving image at gold-standard
        moving_image: torch.Tensor | Error = data_manager().get("moving_images")
        if isinstance(moving_image, Error):
            raise RuntimeError(f"Error getting moving image: {moving_image.description}")
        axes[2].imshow(moving_image[0].cpu().numpy())
        axes[2].set_title("moving image at G.T.")

        plt.ion()  # figures are non-blocking
        plt.show()

    # -----
    # Running repeated registrations with configured parameters
    dimensionality = 6
    distance_samples = torch.empty([int(config.sample_count_per_distance), int(config.reg_config.iteration_count)],
                                   dtype=torch.float64, device=device)  # size = (sample count, iteration count)
    transformation_gt: Transformation | None | Error = data_manager().get("transformation_gt")
    if isinstance(transformation_gt, Error):
        raise Exception(f"Failed to get ground truth transformation: {transformation_gt.description}")
    if transformation_gt is None:
        raise Exception(f"No ground truth transformation available.")
    for i in tqdm(  #
            range(1 if plot else int(config.sample_count_per_distance)),  #
            desc=indentation_prefix(tqdm_position) + "Repeated samples",  #
            position=tqdm_position,  #
            leave=None  #
    ):
        starting_tr = transformation_gt.with_random_offset_at_distance(config.starting_distance)
        starting_params = mapping_transformation_to_parameters(starting_tr)
        # -----
        # Registration
        res = run_reg(  #
            # obj_fun=objective_function if batch_size == 1 else objective_function_batched,  #
            obj_fun=new_objective_function,  #
            # obj_fun=new_new_objective_function,  #
            config=config.reg_config,  #
            starting_params=starting_params,  #
            device=device,  #
            tqdm_position=tqdm_position + 1,  #
            batch_size=batch_size,  #
            plot=plot,  #
            periodic_behaviour=periodic_behaviour,  #
            dry_run=dry_run,  #
        )  # size = (iteration count, dimensionality + 1)
        if not dry_run:
            distance_samples[i, :] = torch.tensor([  #
                transformation_gt.distance(mapping_parameters_to_transformation(row))  #
                for row in res[:, 0:dimensionality]  #
            ], device=distance_samples.device, dtype=distance_samples.dtype)  # size = (iteration count,)

    if plot:
        fig, axes = plt.subplots()
        axes.plot(range(config.reg_config.iteration_count), distance_samples[0, :].cpu().numpy())
        axes.set_xlabel("iteration")
        axes.set_ylabel("distance from gold standard")
        axes.set_ylim((0.0, None))
        plt.draw()
        plt.pause(0.1)
        plt.ioff()  # figures are blocking
        plt.show()

    return None if (dry_run or plot) else pd.DataFrame({  #
        "iteration": torch.arange(config.reg_config.iteration_count).numpy(),  # size = (iteration count,)
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
        config = ExperimentConfig(**dict_config_copy)
    except Exception as e:
        return Error(f"Failed to construct ExperimentConfig: {e}\nParameters:\n{pprint.pformat(dict_config_copy)}")

    return config
