import copy
import logging
import pprint
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import torch
import traitlets
from jaxtyping import Float64

from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.experiments.registration import RegistrationConfig, register
from reg23_experiments.experiments.run_experiments import Experiment, ExperimentConfig
from reg23_experiments.ops.data_manager import args_from_dadg, dadg_updater, data_manager
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation, \
    mapping_transformation_to_parameters
from reg23_experiments.utils.console_logging import indentation_prefix, tqdm

from ._dadg_updaters import batched
from ._of_together import objective_function_alpha_weighted, objective_function_binary_weighted, \
    objective_function_together

__all__ = ["ExperimentParametrisation", "reg_experiment"]

logger = logging.getLogger(__name__)


class ExperimentParametrisation(traitlets.HasTraits):
    """
    Notes:
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
    apply_weighting: bool = traitlets.Bool(default_value=traitlets.Undefined)
    weight_alpha: float = traitlets.Float(min=0.0, default_value=traitlets.Undefined)
    iterations_per_weight_update: int = traitlets.Int(min=0,
                                                      default_value=traitlets.Undefined)  # 0 means every o.f. eval.
    sim_metric: Literal["zncc", "gradient_correlation", "gradient_difference", "mutual_information"] = traitlets.Enum(
        values=[  #
            "zncc",  #
            "gradient_correlation",  #
            "gradient_difference",  #
            "mutual_information",  #
        ], default_value=traitlets.Undefined)
    # ----- registration
    starting_distance: float = traitlets.Float(default_value=traitlets.Undefined)
    sample_count_per_distance: int = traitlets.Int(min=1, default_value=traitlets.Undefined)
    reg_config: RegistrationConfig = traitlets.Instance(RegistrationConfig, allow_none=False, default_value=traitlets.Undefined)

    @staticmethod
    def dict_constructor(dict_config: dict[str, Any]) -> 'ExperimentParametrisation | Error':
        dict_config_copy = copy.deepcopy(dict_config)
        try:
            reg_config = RegistrationConfig(  #
                particle_count=dict_config_copy.pop("particle_count"),  #
                particle_initialisation_spread=dict_config_copy.pop("particle_initialisation_spread"),  #
                iteration_count=dict_config_copy.pop("iteration_count")  #
            )
        except Exception as e:
            return Error(f"Failed to construct RegConfig: {e}\nParameters:\n{pprint.pformat(dict_config_copy)}")

        dict_config_copy["reg_config"] = reg_config

        try:
            config = ExperimentParametrisation(**dict_config_copy)
        except Exception as e:
            return Error(f"Failed to construct ExperimentConfig: {e}\nParameters:\n{pprint.pformat(dict_config_copy)}")

        return config


def reg_experiment(  #
        config: ExperimentConfig[ExperimentParametrisation],  #
        *,  #
        batch_size: int = 1,  #
        plot: bool = False,  #
) -> pd.DataFrame | None:
    """
    Run multiple (`sample_count_per_distance`) registrations according to the given parameters, and return the average
    distance from ground truth at each iteration.
    :param config:
    :param batch_size:
    :param plot:
    :return: A tensor of size (iteration count,) or None; the distance from g.t. of the optimisation at each
    iteration, averaged over `sample_count_per_distance` repetitions, unless the configuration is trivial /
    unnecessary, in which case `None`.
    """
    params = config.params

    data_manager().set("ct_path", params.ct_path, check_equality=True)
    data_manager().set("xray_path", params.xray_path, check_equality=True)
    data_manager().set("downsample_level", params.downsample_level, check_equality=True)
    data_manager().set("truncation_percent", params.truncation_percent, check_equality=True)
    # -----
    # Configuring according to desired similarity metric
    # p_sim_met: ParametrisedSimilarityMetric = string_to_sim_met(params.sim_metric)
    data_manager().set("cropping_method", params.cropping_method, check_equality=True)
    data_manager().set("crop_min_size", params.crop_min_size, check_equality=True)
    data_manager().set("apply_scaling", params.apply_scaling, check_equality=True)
    data_manager().set("apply_weighting", params.apply_weighting, check_equality=True)
    data_manager().set("weight_alpha", params.weight_alpha, check_equality=True)
    data_manager().set("sim_metric", params.sim_metric, check_equality=True)

    # -----
    # Defining the objective function
    def objective_function(parameters: Float64[torch.Tensor, "b 6"]) -> Float64[torch.Tensor, "b"]:
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
    if params.iterations_per_crop_update > 0:
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

        periodic_behaviour.append((params.iterations_per_crop_update, do_crop_refresh))
    else:
        batch_size = 1  # necessary for crop refresh every o.f. evaluation
        data_manager().add_updater(  #
            "refresh_cropping",  #
            dadg_updater(names_returned=["further_cropping"])(batched.refresh_cropping),  #
        )
    # Weight image updates, only after the crop updates
    if params.iterations_per_weight_update > 0:
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

        periodic_behaviour.append((params.iterations_per_weight_update, do_weight_refresh))
    else:
        data_manager().add_updater(  #
            "refresh_weights",  #
            dadg_updater(names_returned=["weight_images"])(batched.refresh_weights),  #
        )

    if plot:
        # Show the target image, fixed image and moving image at the gold-standard
        fig, axes = plt.subplots(1, 5)

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
        # Scaling image at gold-standard
        scaling_image: torch.Tensor | Error = data_manager().get("scaling_images")
        if isinstance(scaling_image, Error):
            raise RuntimeError(f"Error getting fixed image: {scaling_image.description}")
        axes[1].imshow(scaling_image[0].cpu().numpy())
        axes[1].set_title("scaling image")
        # Weighting image at gold-standard
        weighting_image: torch.Tensor | None | Error = data_manager().get("weight_images")
        if isinstance(weighting_image, Error):
            raise RuntimeError(f"Error getting fixed image: {weighting_image.description}")
        if weighting_image is not None:
            axes[2].imshow(weighting_image[0].cpu().numpy())
            axes[2].set_title("weighting image")
        # Fixed image at gold-standard
        fixed_image: torch.Tensor | Error = data_manager().get("fixed_images")
        if isinstance(fixed_image, Error):
            raise RuntimeError(f"Error getting fixed image: {fixed_image.description}")
        axes[3].imshow(fixed_image[0].cpu().numpy())
        axes[3].set_title("fixed image")
        # Moving image at gold-standard
        moving_image: torch.Tensor | Error = data_manager().get("moving_images")
        if isinstance(moving_image, Error):
            raise RuntimeError(f"Error getting moving image: {moving_image.description}")
        axes[4].imshow(moving_image[0].cpu().numpy())
        axes[4].set_title("moving image at G.T.")

        plt.ion()  # figures are non-blocking
        plt.show()

    # -----
    # Running repeated registrations with configured parameters
    dimensionality = 6
    distance_samples = torch.empty([int(params.sample_count_per_distance), int(params.reg_config.iteration_count)],
                                   dtype=torch.float64, device=config.device)  # size = (sample count, iteration count)
    transformation_gt: Transformation | None | Error = data_manager().get("transformation_gt")
    if isinstance(transformation_gt, Error):
        raise Exception(f"Failed to get ground truth transformation: {transformation_gt.description}")
    if transformation_gt is None:
        raise Exception(f"No ground truth transformation available.")
    for i in tqdm(  #
            range(1 if plot else int(params.sample_count_per_distance)),  #
            desc=indentation_prefix(config.tqdm_position) + "Repeated samples",  #
            position=config.tqdm_position,  #
            leave=None  #
    ):
        starting_tr = transformation_gt.with_random_offset_at_distance(params.starting_distance)
        starting_params = mapping_transformation_to_parameters(starting_tr)
        # -----
        # Registration
        res = register(  #
            # obj_fun=objective_function if batch_size == 1 else objective_function_batched,  #
            obj_fun=objective_function,  #
            # obj_fun=new_new_objective_function,  #
            config=params.reg_config,  #
            starting_params=starting_params,  #
            device=config.device,  #
            tqdm_position=config.tqdm_position + 1,  #
            batch_size=batch_size,  #
            plot=plot,  #
            periodic_behaviour=periodic_behaviour,  #
            dry_run=config.dry_run,  #
        )  # size = (iteration count, dimensionality + 1)
        if not config.dry_run:
            distance_samples[i, :] = torch.tensor([  #
                transformation_gt.distance(mapping_parameters_to_transformation(row))  #
                for row in res[:, 0:dimensionality]  #
            ], device=distance_samples.device, dtype=distance_samples.dtype)  # size = (iteration count,)

    if plot:
        fig, axes = plt.subplots()
        axes.plot(range(params.reg_config.iteration_count), distance_samples[0, :].cpu().numpy())
        axes.set_xlabel("iteration")
        axes.set_ylabel("distance from gold standard")
        axes.set_ylim((0.0, None))
        plt.draw()
        plt.pause(0.1)
        plt.ioff()  # figures are blocking
        plt.show()

    return None if (config.dry_run or plot) else pd.DataFrame({  #
        "iteration": torch.arange(params.reg_config.iteration_count).numpy(),  # size = (iteration count,)
        "distance": distance_samples.mean(dim=0).cpu().numpy(),  # size = (iteration count,)
        "distance_std": distance_samples.std(dim=0).cpu().numpy(),  #
    })


# Check that the `reg_experiment` function adheres to the Experiment type
_: Experiment[ExperimentParametrisation] = reg_experiment
