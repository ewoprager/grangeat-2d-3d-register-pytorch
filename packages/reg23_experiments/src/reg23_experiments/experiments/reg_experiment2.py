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

from reg23_experiments.data.structs import Cropping, Error, Transformation
from reg23_experiments.experiments.batched import objective_function_alpha_weighted, \
    objective_function_binary_weighted, objective_function_together
from reg23_experiments.experiments.dadg_updaters import batched
from reg23_experiments.experiments.registration import RegConfig, run_reg
from reg23_experiments.ops import geometry
from reg23_experiments.ops.data_manager import args_from_dadg, dadg_updater, data_manager
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation, \
    mapping_transformation_to_parameters
from reg23_experiments.utils.console_logging import tqdm

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
    iterations_per_crop_update: int = traitlets.Int(min=0, default_value=traitlets.Undefined)  # 0 means every o.f. eval.
    # ----- scaling
    apply_scaling: bool = traitlets.Bool(default_value=traitlets.Undefined)
    # ----- similarity & weighting
    weighting: None | Literal["linear"] | float = traitlets.Union(trait_types=[  #
        traitlets.Enum(values=["linear"], default_value=traitlets.Undefined),  #
        traitlets.Float(min=0.0, default_value=traitlets.Undefined),  #
    ], allow_none=True, default_value=traitlets.Undefined)
    iterations_per_weight_update: int = traitlets.Int(min=0, default_value=traitlets.Undefined)  # 0 means every o.f. eval.
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
        plot: Literal["no", "yes", "mask"] = "no",  #
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
    data_manager().set("cropping_method", config.cropping_method)
    data_manager().set("apply_scaling", config.apply_scaling)
    data_manager().set("weighting", config.weighting)
    data_manager().set("sim_metric", config.sim_metric)

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
    # Weight image updates
    if config.iterations_per_weight_update > 0:
        data_manager().remove_updater("refresh_weights")

        def do_weight_refresh(best_parameters: Float64[torch.Tensor, "6"]) -> None:
            # get the scaling image at the best transformation
            value_dict = args_from_dadg(names_left=["parameters"])(batched.refresh_scaling_images)(
                parameters=best_parameters.unsqueeze(0))
            best_scaling_image = value_dict["scaling_images"]
            # convert to weight image
            value_dict = args_from_dadg(names_left=["scaling_images"])(batched.refresh_weights)(
                scaling_images=best_scaling_image)
            weight_image = value_dict["weight_images"]
            # set the weight image
            data_manager().set("weight_images", weight_image)

        periodic_behaviour.append((config.iterations_per_weight_update, do_weight_refresh))
    else:
        data_manager().add_updater(  #
            "refresh_weights",  #
            dadg_updater(names_returned=["weight_images"])(batched.refresh_weights),  #
        )
    # Crop rectangle updates
    if config.iterations_per_crop_update > 0:
        data_manager().remove_updater("refresh_cropping")

        def do_crop_refresh(best_parameters: Float64[torch.Tensor, "6"]) -> None:
            # get the cropping at the best transformation
            value_dict = args_from_dadg(names_left=["parameters"])(batched.refresh_cropping)(
                parameters=best_parameters.unsqueeze(0))
            best_cropping = value_dict["further_cropping"]
            # set the cropping
            data_manager().set("further_cropping", best_cropping)

        periodic_behaviour.append((config.iterations_per_crop_update, do_crop_refresh))
    else:
        batch_size = 1 # necessary for crop refresh every o.f. evaluation
        data_manager().add_updater(  #
            "refresh_cropping",  #
            dadg_updater(names_returned=["further_cropping"])(batched.refresh_cropping),  #
        )

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
            range(int(config.sample_count_per_distance) if plot == "no" else 1),  #
            desc="Repeated samples",  #
            position=tqdm_position,  #
            leave=None  #
    ):
        starting_tr = transformation_gt.with_random_offset_at_distance(config.starting_distance)
        starting_params = mapping_transformation_to_parameters(starting_tr)
        # -----
        # Crop to the non-zero domain of the DRR at the starting parameters
        data_manager().set("current_transformation", starting_tr)
        cropping: Cropping = args_from_dadg()(geometry.get_crop_nonzero_drr)()
        if cropping.is_collapsed(config.crop_min_size):
            cropping = cropping.uncollapse(config.crop_min_size)
        data_manager().set("further_cropping", cropping, check_equality=True)
        # -----
        # Plotting if desired
        if plot != "no":
            plt.ion()  # figures are non-blocking
            plt.show()
            fig, axes = plt.subplots(1, 3)
            # Getting the data from the DADG
            image_2d_full: torch.Tensor | Error = data_manager().get("image_2d_full")
            if isinstance(image_2d_full, Error):
                raise RuntimeError(f"Error getting image_2d_full: {image_2d_full.description}")
            cropped_target: torch.Tensor | Error = data_manager().get("cropped_target")
            if isinstance(cropped_target, Error):
                raise RuntimeError(f"Error getting fixed image: {cropped_target.description}")
            # Full 2D image
            axes[0].imshow(image_2d_full.cpu().numpy())
            axes[0].set_title("full 2d image")
            # Cropped target
            axes[1].imshow(cropped_target.cpu().numpy())
            axes[1].set_title("cropped target at start")
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
            dry_run=dry_run,#
        )  # size = (iteration count, dimensionality + 1)
        if not dry_run:
            distance_samples[i, :] = torch.tensor([  #
                transformation_gt.distance(mapping_parameters_to_transformation(row))  #
                for row in res[:, 0:dimensionality]  #
            ], device=distance_samples.device, dtype=distance_samples.dtype)  # size = (iteration count,)

    if plot != "no":
        axes[2].plot(range(config.reg_config.iteration_count), distance_samples[0, :].cpu().numpy())
        axes[2].set_xlabel("iteration")
        axes[2].set_ylabel("distance from gold standard")
        axes[2].set_ylim((0.0, None))
        plt.ioff()  # figures are blocking
        plt.show()

    return None if (dry_run or plot != "no") else pd.DataFrame({  #
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
