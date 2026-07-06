import argparse
import copy
import itertools
import os
import pathlib
import pprint
import types
from datetime import datetime
from typing import Any, Callable, Literal, Sequence

import matplotlib
import yaml

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import traitlets
import SimpleITK as sitk

from reg23_experiments.data.structs import Error, Transformation, Cropping
from reg23_experiments.data.transformation_save_data import TransformationSaveData
from reg23_experiments.data.xray_reg_save_data import XRayRegSaveData
from reg23_experiments.experiments import multi_xray_truncation_updaters, updaters
from reg23_experiments.io.image import XrayDICOM, load_cached_drr, read_dicom
from reg23_experiments.io.save_data import load_latest_save
from reg23_experiments.io.command_line import get_string_required
from reg23_experiments.io.serialize import serialize_recursive
from reg23_experiments.io.sitk import find_ct_series, load_ct_series, DCMSeriesInfo
from reg23_experiments.ops import drr, geometry, similarity_metric, swarm as pso
from reg23_experiments.ops.ct import convert_ct_to_mu_sitk
from reg23_experiments.ops.data_manager import args_from_dadg, dadg_updater, data_manager
from reg23_experiments.ops.objective_function import ParametrisedSimilarityMetric
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation, \
    mapping_transformation_to_parameters, random_parameters_at_distance
from reg23_experiments.ops.volume import downsample_trilinear_antialiased
from reg23_experiments.utils import logs_setup, pushover
from reg23_experiments.utils.console_logging import tqdm


def configs_to_dict(*vargs) -> dict[str, Any]:
    # convert all function pointers to their `str` names and merge all configs
    return {k: (v.__qualname__ if isinstance(v, types.FunctionType) else v) for config in vargs for k, v in
            config.trait_values().items()}


def save_dict(d: dict, *, directory: pathlib.Path, stem: str) -> None:
    directory.mkdir(exist_ok=True, parents=True)
    torch.save(d, directory / f"{stem}.pkl")
    (directory / f"{stem}.txt").write_text(pprint.pformat(d))


def instance_output_directory(script_output_directory: str | pathlib.Path) -> pathlib.Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ret: pathlib.Path = pathlib.Path(script_output_directory) / timestamp
    ret.mkdir(parents=True, exist_ok=True)
    return ret


def load_untruncated_ct(  #
        ct_path: pathlib.Path,  #
        device: torch.device,  #
        ct_permutation: Sequence[int] | None = None  #
) -> tuple[torch.Tensor, torch.Tensor, str] | Error:
    series: dict[str, DCMSeriesInfo] | Error = find_ct_series(ct_path)
    if isinstance(series, Error):
        raise Exception(f"Failed to open CT from path '{ct_path}': {series.description}")
    if not series:
        return Error(f"No CT series found at path '{str(ct_path)}'.")
    if len(series) == 1:
        key = next(iter(series))
    else:
        key = get_string_required(  #
            f"Please choose one of the following CT series:\n"
            f"{"\n".join(f"{k}:\n\t{pprint.pformat(serialize_recursive(v))}\n" for k, v in series.items())}",  #
            lambda k: None if k in series else Error(f"String '{k}' does not name a series.")  #
        )
    volume: sitk.Image | Error = load_ct_series(ct_path, key)
    if isinstance(volume, Error):
        return Error(f"Failed to open CT from path '{str(ct_path)}': {volume.description}")
    tensor: torch.Tensor | Error = convert_ct_to_mu_sitk(volume, dtype=torch.float32)
    if isinstance(tensor, Error):
        return Error(f"Failed to convert CT from path '{str(ct_path)}' to mu: {tensor.description}")
    tensor = tensor.to(device=device)
    spacing = torch.tensor(volume.GetSpacing(), device=device, dtype=torch.float64)
    if ct_permutation is not None:
        if len(ct_permutation) != 3:
            return Error("Length of ct_permutation must be 3.")
        tensor = tensor.permute(*ct_permutation)
        spacing = spacing[torch.tensor(ct_permutation)]

    return tensor, spacing, key


@dadg_updater(names_returned=["source_distance", "image_2d_full", "image_2d_full_spacing", "transformation_gt"])
def set_synthetic_target_image(  #
        *,  #
        ct_path: str,  #
        ct_spacing: torch.Tensor,  #
        untruncated_ct_volume: torch.Tensor,  #
        new_drr_size: torch.Size,  #
        regenerate_drr: bool,  #
        save_to_cache: bool,  #
        cache_directory: str,  #
        ap_transformation: Transformation,  #
        target_ap_distance: float  #
) -> dict[str, Any]:
    # generate a DRR through the volume
    drr_spec = None
    if not regenerate_drr:
        drr_spec = load_cached_drr(cache_directory, ct_path)

    if drr_spec is None:
        tr = mapping_parameters_to_transformation(
            random_parameters_at_distance(mapping_transformation_to_parameters(ap_transformation), target_ap_distance))
        drr_spec = drr.generate_drr_as_target(cache_directory, ct_path, untruncated_ct_volume, ct_spacing,
                                              save_to_cache=save_to_cache, size=new_drr_size, transformation=tr)

    image_2d_full_spacing, scene_geometry, image_2d_full, transformation_ground_truth = drr_spec
    del drr_spec

    return {"source_distance": scene_geometry.source_distance, "image_2d_full": image_2d_full,
            "image_2d_full_spacing": image_2d_full_spacing, "transformation_gt": transformation_ground_truth}


@dadg_updater(names_returned=["source_distance", "image_2d_full", "image_2d_full_spacing", "xray_sop_instance_uid"])
def set_xray_target_image(*, xray_path: str, device: torch.device) -> dict[str, Any]:
    dicom: XrayDICOM = read_dicom(xray_path)
    image_2d_full = dicom["image"].to(device=device, dtype=torch.float32)
    image_2d_full_spacing = dicom["spacing"].to(device=device, dtype=torch.float64)
    return {  #
        "source_distance": dicom["scene_geometry"].source_distance,  ##
        "image_2d_full": image_2d_full,  #
        "image_2d_full_spacing": image_2d_full_spacing,  #
        "xray_sop_instance_uid": dicom["uid"]  #
    }


@dadg_updater(names_returned=["transformation_gt"])
def load_ground_truth(  #
        *,  #
        saved_transformations: pd.DataFrame,  #
        xray_sop_instance_uid: str,  #
        device: torch.device  #
) -> dict[str, Any]:
    idx = (xray_sop_instance_uid, "gold_standard")
    try:
        row = saved_transformations.loc[idx]
    except KeyError:
        return {"transformation_gt": None}
    return {  #
        "transformation_gt": Transformation.from_vector(  #
            torch.tensor([row[f"x{i}"] for i in range(6)], device=device, dtype=torch.float64)  #
        )  #
    }


@dadg_updater(names_returned=["base_cropping"])
def load_base_cropping(  #
        *,  #
        saved_xray_reg_configs: pd.DataFrame,  #
        xray_sop_instance_uid: str,  #
) -> dict[str, Any]:
    try:
        row = saved_xray_reg_configs.loc[xray_sop_instance_uid]
    except KeyError:
        return {"base_cropping": None}
    return {  #
        "base_cropping": Cropping(  #
            left=row["crop_left"],  #
            right=row["crop_right"],  #
            top=row["crop_top"],  #
            bottom=row["crop_bottom"],  #
        )  #
    }


@dadg_updater(names_returned=["cropping"])
def combine_croppings(  #
        *,  #
        base_cropping: Cropping | None,  #
        further_cropping: Cropping | None,  #
) -> dict[str, Any]:
    if base_cropping is None:
        if further_cropping is None:
            return {"cropping": None}
        else:
            return {"cropping": further_cropping}
    else:
        if further_cropping is None:
            return {"cropping": base_cropping}
        else:
            return {"cropping": Cropping.intersect(base_cropping, further_cropping)}


@dadg_updater(names_returned=["ct_volumes"])
def apply_truncation(*, untruncated_ct_volume: torch.Tensor, truncation_percent: int) -> dict[str, Any]:
    # truncate the volume
    truncation_fraction = 0.01 * float(truncation_percent)
    top_bottom_chop = int(round(0.5 * truncation_fraction * float(untruncated_ct_volume.size()[0])))
    ct_volume = untruncated_ct_volume[
        top_bottom_chop:max(top_bottom_chop + 1, untruncated_ct_volume.size()[0] - top_bottom_chop)]
    # mipmap the volume
    ct_volumes = [ct_volume]
    level: int = 1
    while torch.tensor(ct_volumes[-1].size()).min() > 3:
        ct_volumes.append(downsample_trilinear_antialiased(ct_volumes[0], scale_factor=0.5 ** float(level)))
        level += 1
    return {"ct_volumes": ct_volumes}


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
        tqdm_position: int = 0  #
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
        device=device  #
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


class ExperimentConfig(traitlets.HasTraits):
    ct_path: str = traitlets.Unicode(default_value=traitlets.Undefined)
    xray_path: str = traitlets.Unicode(default_value=traitlets.Undefined)
    downsample_level: int = traitlets.Int(min=0, default_value=traitlets.Undefined)
    truncation_percent: int = traitlets.Int(min=0, max=100, default_value=traitlets.Undefined)
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


def string_to_sim_met(  #
        config_string: str,  #
        *,  #
        kernel_size: int = 8,  #
        llambda: float = 1.0,  #
        gradient_method: Literal["sobel", "central_difference"] = "sobel"  #
) -> ParametrisedSimilarityMetric:
    if config_string == "zncc":
        return ParametrisedSimilarityMetric(similarity_metric.ncc)
    elif config_string == "local_zncc":
        return ParametrisedSimilarityMetric(similarity_metric.local_ncc, kernel_size=kernel_size)
    elif config_string == "multiscale_zncc":
        return ParametrisedSimilarityMetric(similarity_metric.multiscale_ncc, kernel_size=kernel_size, llambda=llambda)
    elif config_string == "gradient_correlation":
        return ParametrisedSimilarityMetric(similarity_metric.gradient_correlation, gradient_method=gradient_method)
    raise ValueError(f"Unknown similarity metric '{config_string}'.")


def run_experiment(  #
        *,  #
        reg_config: RegConfig,  #
        exp_config: ExperimentConfig,  #
        device: torch.device,  #
        tqdm_position: int = 0  #
) -> pd.DataFrame | None:
    """
    Run multiple (`sample_count_per_distance`) registrations according to the given parameters, and return the average
    distance from ground truth at each iteration.
    :param reg_config:
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
    data_manager().set("truncation_percent", exp_config.truncation_percent, check_equality=True)
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
    def objective_function(parameters: torch.Tensor) -> torch.Tensor:
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
    distance_samples = torch.empty([int(exp_config.sample_count_per_distance), int(reg_config.iteration_count)],
                                   dtype=torch.float64, device=device)  # size = (sample count, iteration count)
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
        res = run_reg(  #
            obj_fun=objective_function,  #
            config=reg_config,  #
            starting_params=starting_params,  #
            device=device,  #
            tqdm_position=tqdm_position + 1)  # size = (iteration count, dimensionality + 1)
        distance_samples[i, :] = torch.linalg.vector_norm(res[:, 0:dimensionality] - ground_truth,
                                                          dim=1)  # size = (iteration count,)

    return pd.DataFrame({  #
        "iteration": torch.arange(reg_config.iteration_count).numpy(),  # size = (iteration count,)
        "distance": distance_samples.mean(dim=0).cpu().numpy(),  # size = (iteration count,)
        "distance_std": distance_samples.std(dim=0).cpu().numpy(),  #
        "crop_width": crop_size_samples[:, 0].mean().cpu().numpy(),  #
        "crop_height": crop_size_samples[:, 1].mean().cpu().numpy(),  #
    })


def run_experiments(  #
        *,  #
        params_to_vary: dict[str, list | torch.Tensor],  #
        constants: dict[str, Any],  #
        output_directory: pathlib.Path,  #
        device: torch.device,  #
        tqdm_position: int = 0  #
) -> None:
    assert output_directory.is_dir()
    # -----
    # Determine the total number of experiments being run
    each_range_length = []
    for name, values in params_to_vary.items():
        if isinstance(values, torch.Tensor):
            assert len(values.size()) == 1
        each_range_length.append(len(values))
    total = 1
    for l in each_range_length:
        total *= l
    logger.info(f"Running experiments with the following constant parameters:\n{pprint.pformat(constants)}")
    # -----
    # Iterate through the Cartesian product of the sets of parameters values, and run an experiment for each
    tqdm_iterator = tqdm(  #
        itertools.product(*(range(l) for l in each_range_length)),  #
        desc="Experiments",  #
        total=total,  #
        position=tqdm_position,  #
        leave=None  #
    )
    for indices in tqdm_iterator:
        # -----
        # Unpack the parameters for this iteration
        instance_specific: dict[str, Any] = {  #
            name: values[index]  #
            for index, (name, values) in zip(indices, params_to_vary.items())  #
        }  # config specific to this instance
        tqdm_iterator.set_postfix(**instance_specific)  # displaying this
        instance_all: dict[str, Any] = instance_specific | constants  # all the config for this instance
        # -----
        # Separate the config into registration and experiment configs
        exp_config_by_name = copy.deepcopy(instance_all)
        try:
            reg_config = RegConfig(  #
                particle_count=exp_config_by_name.pop("particle_count"),  #
                particle_initialisation_spread=exp_config_by_name.pop("particle_initialisation_spread"),  #
                iteration_count=exp_config_by_name.pop("iteration_count")  #
            )
        except Exception as e:
            logger.error(f"Error constructing registration configuration at indices {indices}: {e}")
            continue
        try:
            exp_config = ExperimentConfig(**exp_config_by_name)
        except Exception as e:
            logger.error(f"Error constructing experiment configuration at indices {indices}: {e}\nParameters:\n"
                         f"{pprint.pformat(instance_all)}")
            continue
        # -----
        # Run the experiment
        try:
            res: pd.DataFrame | None = run_experiment(reg_config=reg_config, exp_config=exp_config, device=device,
                                                      tqdm_position=tqdm_position + 1)
        except Exception as e:
            logger.error(
                f"Error running experiment at indices {indices}: {e}\nParameters:\n{pprint.pformat(instance_all)}")
            continue
        if res is None:
            logger.info(
                f"Experiment at indices {indices}; configuration: \n{pprint.pformat(instance_specific)}\nwas deemed "
                f"trivial / unnecessary.")
            continue
        # -----
        # Add the experiment config rows to the DataFrame and save
        df = res.assign(**instance_all)
        df.to_parquet(output_directory / f"data_{"_".join([str(i) for i in indices])}.parquet")


def main(  #
        *,  #
        cache_directory: str,  #
        ct_path: str,  #
        xray_path: str | pathlib.Path | None,  #
        data_output_dir: str | pathlib.Path,  #
        show: bool = False  #
):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if xray_path is not None:
        xray_path = pathlib.Path(xray_path)

    # -----
    # Load the CT data, prompting the user to choose a series if multiple are found
    res = load_untruncated_ct(pathlib.Path(ct_path), device)
    if isinstance(res, Error):
        raise Exception(f"Failed to load CT: {res.description}")
    untruncated_ct_volume, ct_spacing, ct_series_uid = res

    # -----
    # Load all saved transformations; these are searched through for ground truth alignments
    res: tuple[pathlib.Path, TransformationSaveData, int] | Error = load_latest_save(  #
        TransformationSaveData,  #
        save_directory=pathlib.Path("data/app_transformation_save_data")  #
    )
    if isinstance(res, Error):
        raise RuntimeError(f"Failed to load saved transformation: {res.description}")
    _, transformation_save_data, _ = res
    saved_transformations: pd.DataFrame = transformation_save_data.get_data()
    logger.info(f"Saved transformation data:\n{saved_transformations.to_string()}")

    # -----
    # Load all saved X-ray configs; these are used for manual X-ray configurations
    res: tuple[pathlib.Path, XRayRegSaveData, int] | Error = load_latest_save(  #
        XRayRegSaveData,  #
        save_directory=pathlib.Path("data/xray_reg_save_data")  #
    )
    if isinstance(res, Error):
        raise RuntimeError(f"Failed to load saved X-ray reg configs: {res.description}")
    _, xray_reg_save_data, _ = res
    saved_xray_reg_configs: pd.DataFrame = xray_reg_save_data.get_data()
    logger.info(f"Saved X-ray reg configs:\n{saved_xray_reg_configs.to_string()}")

    # -----
    # Initialise the DADG
    if isinstance(err := data_manager().set_multiple(  #
            device=device,  #
            untruncated_ct_volume=untruncated_ct_volume,  #
            ct_spacing=ct_spacing,  #
            ct_series_uid=ct_series_uid,  #
            cache_directory=cache_directory,  #
            save_to_cache=False,  #
            truncation_percent=0,  #
            further_cropping=None,  #
            source_offset=torch.zeros(2, dtype=torch.float64, device=device),  #
            downsample_level=0,  #
            ap_transformation=Transformation(
                rotation=torch.tensor([0.5 * torch.pi, 0.0, 0.0], dtype=torch.float64, device=device),
                translation=torch.zeros(3, dtype=torch.float64, device=device)),  #
            target_ap_distance=5.0,  #
            current_transformation=Transformation.random_uniform(device=device),  #
            mask_transformation=None,  #
            saved_transformations=saved_transformations,  #
            saved_xray_reg_configs=saved_xray_reg_configs,  #
    ), Error):
        logger.error(f"Error setting initial data values: {err.description}")
        return

    # -----
    # Initialise the fixed target image
    if xray_path is None:
        # -----
        # Use a DRR
        if isinstance(err := data_manager().set_multiple(  #
                xray_path=None,  #
                regenerate_drr=True,  #
                new_drr_size=torch.Size([1000, 1000]),  #
                target_ap_distance=5.0,  #
        ), Error):
            logger.error(f"Error setting initial data values: {err.description}")
            return

        if isinstance(err := data_manager().add_updater("set_target_image", set_synthetic_target_image), Error):
            logger.error(f"Error adding updater: {err.description}")
            return
    elif xray_path.is_dir():
        # -----
        # Use a directory of X-ray images
        if isinstance(err := data_manager().add_updater("set_target_image", set_xray_target_image), Error):
            logger.error(f"Error adding updater: {err.description}")
            return

        if isinstance(err := data_manager().add_updater("set_ground_truth", load_ground_truth), Error):
            logger.error(f"Error adding updater: {err.description}")
            return
    else:
        # -----
        # Use an X-ray image
        if not xray_path.is_file():
            raise Exception(f"X-ray file '{str(xray_path)}' not found.")

        if isinstance(err := data_manager().set("xray_path", xray_path), Error):
            logger.error(f"Error setting initial data values: {err.description}")
            return
        if isinstance(err := data_manager().add_updater("set_target_image", set_xray_target_image), Error):
            logger.error(f"Error adding updater: {err.description}")
            return
        if isinstance(err := data_manager().add_updater("set_ground_truth", load_ground_truth), Error):
            logger.error(f"Error adding updater: {err.description}")
            return

    # -----
    # Add updaters to the DADG
    if isinstance(err := data_manager().add_updater("apply_truncation", apply_truncation), Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    if isinstance(err := data_manager().add_updater(  #
            "refresh_image_2d_scale_factor", updaters.refresh_image_2d_scale_factor), Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    if isinstance(err := data_manager().add_updater("refresh_hyperparameter_dependent",
                                                    updaters.refresh_hyperparameter_dependent), Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    if isinstance(err := data_manager().add_updater("refresh_mask_transformation_dependent",
                                                    updaters.refresh_mask_transformation_dependent), Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    if isinstance(err := data_manager().add_updater("project_drr", multi_xray_truncation_updaters.project_drr), Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    if isinstance(err := data_manager().add_updater("load_base_cropping", load_base_cropping), Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    if isinstance(err := data_manager().add_updater("combine_croppings", combine_croppings), Error):
        logger.error(f"Error adding updater: {err.description}")
        return

    # ----------------------------------
    # - Hardcoded script configuration -
    # ----------------------------------
    constants: dict[str, Any] = {  #
        # ExperimentConfig
        "ct_path": ct_path,  #
        "xray_path": xray_path,  #
        "ct_series_uid": data_manager().get("ct_series_uid"),  #
        "downsample_level": 1,  #
        "truncation_percent": 80,  #
        "cropping": "nonzero_drr",  #
        "crop_expand": 0.0,  #
        "crop_min_size": 0.01,  #
        "mask": "None",  #
        "sim_metric": "zncc",  #
        "starting_distance": 5.0,  #
        "sample_count_per_distance": 10,  #
        # RegConfig
        "particle_count": 2000,  #
        "particle_initialisation_spread": 5.0,  #
        "iteration_count": 6,  #
    }
    hardcoded_xray_names: list[str] = [  #
        "level_000",  #
        # "level_090",  #
        "up_000",  #
        # "up_090",  #
        "down_000",  #
        # "down_090",  #
    ]
    params_to_vary: dict[str, list | torch.Tensor] = {  #
        "truncation_percent": [75, 80, 85],  #
        "cropping": ["nonzero_drr", "full_depth_drr"],  #
        "mask": ["None", "Every evaluation weighting zncc"],  #
        "crop_expand": [-15.0, -5.0, 5.0, 15.0],  #
    }
    # ----------------------------------

    # -----
    # Setting the X-ray path(s) if a directory is passed
    if xray_path is not None and xray_path.is_dir():
        # Check that all X-rays exist, have ground truth transformations available, and have reg configs available
        for name in hardcoded_xray_names:
            path: pathlib.Path = xray_path / name
            if not path.is_file():
                logger.error(f"X-ray file '{str(path)}' doesn't exist.")
                return
            try:
                dicom: XrayDICOM = read_dicom(path)
            except Exception as e:
                logger.error(f"Failed to read X-ray file: {e}")
                return
            idx = (dicom["uid"], "gold_standard")
            try:
                saved_transformations.loc[idx]
            except KeyError:
                logger.error(f"No ground truth saved for X-ray '{str(path)}' with UID '{dicom["uid"]}'.")
                return
            idx = dicom["uid"]
            try:
                saved_xray_reg_configs.loc[idx]
            except KeyError:
                logger.error(f"No reg config saved for X-ray '{str(path)}' with UID '{dicom["uid"]}'.")
                return
        if len(hardcoded_xray_names) == 1:
            constants["xray_path"] = str(xray_path / hardcoded_xray_names[0])
        else:
            params_to_vary["xray_path"] = [str(xray_path / name) for name in hardcoded_xray_names]

    # Remove varying variables from the constants dict
    for key in params_to_vary:
        if key in constants:
            constants.pop(key)

    if show:
        # -----
        # Display images for debugging
        plt.ion()  # figures are non-blocking
        plt.show()
        fig, axes = plt.subplots(1, 4)
        # -----
        # Set the current transformation to the ground truth if it exists
        data_manager().set("xray_path", "/home/eprager/Documents/Datasets/3DP Head 2/X-ray/level_000")

        transformation_gt: Transformation | None | Error = data_manager().get("transformation_gt")
        if transformation_gt is None or isinstance(transformation_gt, Error):
            raise RuntimeError(f"No ground truth available"
                               f"{"." if transformation_gt is None else f": {transformation_gt.description}"}")
        parameters_gt = mapping_transformation_to_parameters(transformation_gt)
        starting_params = random_parameters_at_distance(parameters_gt, constants["starting_distance"])

        data_manager().set("current_transformation", transformation_gt)
        if "downsample_level" in constants:
            data_manager().set("downsample_level", constants["downsample_level"])
        image_2d_full: torch.Tensor | Error = data_manager().get("image_2d_full")
        if isinstance(image_2d_full, Error):
            raise RuntimeError(f"Error getting image_2d_full: {image_2d_full.description}")
        axes[0].imshow(image_2d_full.cpu().numpy())
        axes[0].set_title("original target")
        fixed_image: torch.Tensor | Error = data_manager().get("fixed_image")
        if isinstance(fixed_image, Error):
            raise RuntimeError(f"Error getting fixed image: {fixed_image.description}")
        axes[1].imshow(fixed_image.cpu().numpy())
        axes[1].set_title("fixed image")
        moving_image: torch.Tensor | Error = data_manager().get("moving_image")
        if isinstance(moving_image, Error):
            raise RuntimeError(f"Error getting moving image: {moving_image.description}")
        axes[2].imshow(moving_image.cpu().numpy())
        axes[2].set_title("moving image at G.T.")
        data_manager().set("mask_transformation", data_manager().get("current_transformation"))
        mask: torch.Tensor | Error = data_manager().get("mask")
        if isinstance(mask, Error):
            raise RuntimeError(f"Error getting mask: {mask.description}")
        axes[3].imshow(mask.cpu().numpy())
        axes[3].set_title("mask at G.T.")
        logger.info(f"ZNCC at G.T. with masking = "
                    f"{-similarity_metric.weighted_local_ncc(moving_image, fixed_image, mask, kernel_size=8)}")
        plt.draw()
        plt.pause(0.1)

        data_manager().set("current_transformation", mapping_parameters_to_transformation(starting_params))
        data_manager().set("truncation_percent", 65)
        if "cropping" in constants:
            if constants["cropping"] == "None":
                cropping: Cropping | None = None
            elif constants["cropping"] == "nonzero_drr":
                cropping: Cropping | None = args_from_dadg()(geometry.get_crop_nonzero_drr)()
            elif constants["cropping"] == "full_depth_drr":
                cropping: Cropping | None = args_from_dadg()(geometry.get_crop_full_depth_drr)()
            else:
                raise ValueError(f"Unknown cropping technique '{constants["cropping"]}'.")
            if isinstance(cropping, Error):
                raise RuntimeError(f"Failed to set crop: {cropping.description}")
            data_manager().set("further_cropping", cropping, check_equality=True)

        def objective_function(parameters: torch.Tensor) -> torch.Tensor:
            data_manager().set("current_transformation",
                               mapping_parameters_to_transformation(parameters.to(dtype=torch.float64)))
            _moving_image: torch.Tensor | Error = data_manager().get("moving_image")
            _fixed_image: torch.Tensor | Error = data_manager().get("fixed_image")
            return -similarity_metric.ncc(_moving_image, _fixed_image)

        res: torch.Tensor = run_reg(  #
            obj_fun=objective_function,  #
            starting_params=starting_params, config=RegConfig(  #
                particle_count=constants["particle_count"],  #
                particle_initialisation_spread=constants["particle_initialisation_spread"],  #
                iteration_count=constants["iteration_count"],  #
            ),  #
            device=device,  #
            plot="mask")  # size = (iteration count, dimensionality + 1)
        logger.info(f"Result: {res}")
        plt.ioff()  # figures are blocking
        fig, axes = plt.subplots()
        distances = torch.linalg.vector_norm(res[:, :6] - parameters_gt.unsqueeze(0), dim=1).cpu().numpy()
        axes.plot(distances)
        axes.set_xlabel("iteration")
        axes.set_ylabel("distance from G.T.")
        plt.show()
        return

    instance_output_dir: pathlib.Path = instance_output_directory(data_output_dir)

    with open(instance_output_dir / "variables.txt", 'w') as file:
        yaml.safe_dump({  #
            "constants": constants,  #
            "variables": params_to_vary,  #
        }, file)

    # -----
    # Run experiments, setting the parameters to vary
    run_experiments(params_to_vary=params_to_vary, output_directory=instance_output_dir, constants=constants,
                    device=device)


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-c", "--cache-directory", type=str, default="cache",
                        help="Set the directory where data that is expensive to calculate will be saved. The default "
                             "is 'cache'.")
    parser.add_argument("-p", "--ct-path", type=str, required=True,
                        help="Give a path to a .nrrd file, .nii file or directory of .dcm files containing CT data to "
                             "process. If not provided, some simple synthetic data will be used instead - note that "
                             "in this case, data will not be saved to the cache.")
    parser.add_argument("-x", "--xray-path", type=str, default=None,
                        help="Give a path to a DICOM file containing an X-ray image to register the CT image to. If "
                             "this is provided, the X-ray will by used instead of any DRR.")
    parser.add_argument("-d", "--xray-dir", type=str, default=None,
                        help="Give a path to directory of DICOM X-ray images to register the CT image to. If "
                             "this is provided, the X-rays will by used instead of any DRR.")
    # parser.add_argument("-i", "--no-load", action='store_true',
    #                     help="Do not load any pre-calculated data from the cache.")
    # parser.add_argument(
    #     "-r", "--regenerate-drr", action='store_true',
    #     help="Regenerate the DRR through the 3D data, regardless of whether a DRR has been cached.")
    # parser.add_argument("-n", "--no-save", action='store_true', help="Do not save any data to the cache.")
    parser.add_argument("-n", "--notify", action="store_true", help="Send notification on completion.")
    parser.add_argument("-s", "--show", action="store_true", help="Show images at the G.T. alignment.")
    parser.add_argument("-o", "--data-output-dir", type=str, default="experimental_results/program_truncation",
                        help="Directory in which to save output data.")
    args = parser.parse_args()

    if args.xray_path is None:
        if args.xray_dir is None:
            xray = None
        else:
            xray = pathlib.Path(args.xray_dir)
            if not xray.is_dir():
                logger.error(f"X-ray directory '{str(xray)}' doesn't exist.")
                exit(1)
    else:
        if args.xray_dir is None:
            xray = pathlib.Path(args.xray_path)
            if not xray.is_file():
                logger.error(f"X-ray file '{str(xray)}' doesn't exist.")
                exit(1)
        else:
            logger.error(f"Cannot provide both an X-ray directory and an X-ray file.")
            exit(1)

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    try:
        main(cache_directory=args.cache_directory, ct_path=args.ct_path, xray_path=xray,
             data_output_dir=args.data_output_dir, show=args.show)
        if args.notify:
            pushover.send_notification(__file__, "Script finished.")
    except Exception as e:
        if args.notify:
            pushover.send_notification(__file__, "Script raised exception: {}.".format(e))
        raise e
