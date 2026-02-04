import argparse
import os
from typing import Any, Callable, Literal, Sequence
from datetime import datetime
import types
import pprint
import itertools
import copy

import pathlib

import pandas as pd
import torch
import torchviz
import matplotlib.pyplot as plt
import traitlets

from reg23_experiments.utils.console_logging import tqdm
from reg23_experiments.utils import logs_setup, pushover
from reg23_experiments.ops.data_manager import data_manager, init_data_manager, dag_updater, updaters, args_from_dag
from reg23_experiments.data.structs import Error, Transformation, SceneGeometry, Cropping
from reg23_experiments.utils.data import StrictHasTraits
from reg23_experiments.io.volume import load_volume
from reg23_experiments.io.image import load_cached_drr
from reg23_experiments.ops.optimisation import mapping_transformation_to_parameters, \
    mapping_parameters_to_transformation, random_parameters_at_distance
from reg23_experiments.ops import drr, geometry, similarity_metric, swarm as pso
from reg23_experiments.ops.objective_function import ParametrisedSimilarityMetric


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


@dag_updater(names_returned=["untruncated_ct_volume", "ct_spacing"])
def load_untruncated_ct(ct_path: str, device: torch.device, ct_permutation: Sequence[int] | None = None) -> dict[
    str, Any]:
    ct_volume, ct_spacing = load_volume(pathlib.Path(ct_path))
    ct_volume = ct_volume.to(device=device, dtype=torch.float32)
    ct_spacing = ct_spacing.to(device=device)

    if ct_permutation is not None:
        assert len(ct_permutation) == 3
        ct_volume = ct_volume.permute(*ct_permutation)
        ct_spacing = ct_spacing[torch.tensor(ct_permutation)]

    return {"untruncated_ct_volume": ct_volume, "ct_spacing": ct_spacing}


@dag_updater(names_returned=["source_distance", "image_2d_full", "fixed_image_spacing", "transformation_gt"])
def set_target_image(ct_path: str, ct_spacing: torch.Tensor, untruncated_ct_volume: torch.Tensor,
                     new_drr_size: torch.Size, regenerate_drr: bool, save_to_cache: bool, cache_directory: str,
                     ap_transformation: Transformation, target_ap_distance: float) -> dict[str, Any]:
    # generate a DRR through the volume
    drr_spec = None
    if not regenerate_drr:
        drr_spec = load_cached_drr(cache_directory, ct_path)

    if drr_spec is None:
        tr = mapping_parameters_to_transformation(
            random_parameters_at_distance(mapping_transformation_to_parameters(ap_transformation), target_ap_distance))
        drr_spec = drr.generate_drr_as_target(cache_directory, ct_path, untruncated_ct_volume, ct_spacing,
                                              save_to_cache=save_to_cache, size=new_drr_size, transformation=tr)

    fixed_image_spacing, scene_geometry, image_2d_full, transformation_ground_truth = drr_spec
    del drr_spec

    return {"source_distance": scene_geometry.source_distance, "image_2d_full": image_2d_full,
            "fixed_image_spacing": fixed_image_spacing, "transformation_gt": transformation_ground_truth}


@dag_updater(names_returned=["ct_volumes"])
def apply_truncation(untruncated_ct_volume: torch.Tensor, truncation_percent: int) -> dict[str, Any]:
    # truncate the volume
    truncation_fraction = 0.01 * float(truncation_percent)
    top_bottom_chop = int(round(0.5 * truncation_fraction * float(untruncated_ct_volume.size()[0])))
    ct_volume = untruncated_ct_volume[
        top_bottom_chop:max(top_bottom_chop + 1, untruncated_ct_volume.size()[0] - top_bottom_chop)]
    # mipmap the volume
    down_sampler = torch.nn.AvgPool3d(2)
    ct_volumes = [ct_volume]
    while min(ct_volumes[-1].size()) > 1:
        ct_volumes.append(down_sampler(ct_volumes[-1].unsqueeze(0))[0])
    return {"ct_volumes": ct_volumes}


@dag_updater(names_returned=["moving_image"])
def project_drr(ct_volumes: list[torch.Tensor], ct_spacing: torch.Tensor, current_transformation: Transformation,
                fixed_image_size: torch.Size, source_distance: float, fixed_image_spacing: torch.Tensor,
                downsample_level: int, translation_offset: torch.Tensor, fixed_image_offset: torch.Tensor,
                image_2d_scale_factor: float, device) -> dict[str, Any]:
    # Applying the translation offset
    new_translation = current_transformation.translation + torch.cat(
        (torch.tensor([0.0], device=device, dtype=current_transformation.translation.dtype),
         translation_offset.to(device=current_transformation.device)))
    transformation = Transformation(rotation=current_transformation.rotation, translation=new_translation).to(
        device=device)

    return {"moving_image": geometry.generate_drr(ct_volumes[downsample_level], transformation=transformation,
                                                  voxel_spacing=ct_spacing * 2.0 ** downsample_level,
                                                  detector_spacing=fixed_image_spacing / image_2d_scale_factor,
                                                  scene_geometry=SceneGeometry(source_distance=source_distance,
                                                                               fixed_image_offset=fixed_image_offset),
                                                  output_size=fixed_image_size)}


class RegConfig(StrictHasTraits):
    particle_count: int = traitlets.Int(default_value=traitlets.Undefined)
    particle_initialisation_spread: float = traitlets.Float(default_value=traitlets.Undefined)
    iteration_count: int = traitlets.Int(default_value=traitlets.Undefined)


def run_reg(*, obj_fun: Callable, starting_params: torch.Tensor, config: RegConfig, device: torch.device,
            plot: Literal["no", "yes", "mask"] = "no", tqdm_position: int = 0) -> torch.Tensor:
    """

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
        data_manager().set_data("current_transformation", mapping_parameters_to_transformation(starting_params))
        axes[0].imshow(data_manager().get("moving_image").cpu().numpy())
        plt.draw()
        plt.pause(0.1)

    pso_config = pso.OptimisationConfig(objective_function=obj_fun)
    dimensionality = starting_params.numel()
    # initialise the return tensor
    ret = torch.empty([config.iteration_count, dimensionality + 1], dtype=torch.float32, device=device)
    tqdm_iterator = tqdm(range(config.iteration_count), desc="PSO iterations", position=tqdm_position, leave=None)
    # initialise the swarm, which performs an o.f. evaluation for each particle
    swarm = pso.Swarm(config=pso_config, dimensionality=dimensionality, particle_count=config.particle_count,
                      initialisation_position=starting_params,
                      initialisation_spread=torch.full([dimensionality], config.particle_initialisation_spread),
                      device=device)
    ret[0, 0:dimensionality] = swarm.current_optimum_position.to(dtype=torch.float32, device=device)
    ret[0, -1] = swarm.current_optimum.to(dtype=torch.float32, device=device)
    tqdm_iterator.update()

    for it in range(1, config.iteration_count):
        swarm.iterate()
        ret[it, 0:dimensionality] = swarm.current_optimum_position.to(dtype=torch.float32, device=device)
        ret[it, -1] = swarm.current_optimum.to(dtype=torch.float32, device=device)

        if plot != "no":
            data_manager().set_data("current_transformation",
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


class ExperimentConfig(StrictHasTraits):
    ct_path: str = traitlets.Unicode(default_value=traitlets.Undefined)
    downsample_level: int = traitlets.Int(min=0, default_value=traitlets.Undefined)
    truncation_percent: int = traitlets.Int(min=0, max=100, default_value=traitlets.Undefined)
    cropping: str = traitlets.Enum(values=["None", "nonzero_drr", "full_depth_drr"], default_value=traitlets.Undefined)
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


def string_to_sim_met(config_string: str, *, kernel_size: int = 8, llambda: float = 1.0, gradient_method: Literal[
    "sobel", "central_difference"] = "sobel") -> ParametrisedSimilarityMetric:
    if config_string == "zncc":
        return ParametrisedSimilarityMetric(similarity_metric.ncc)
    elif config_string == "local_zncc":
        return ParametrisedSimilarityMetric(similarity_metric.local_ncc, kernel_size=kernel_size)
    elif config_string == "multiscale_zncc":
        return ParametrisedSimilarityMetric(similarity_metric.multiscale_ncc, kernel_size=kernel_size, llambda=llambda)
    elif config_string == "gradient_correlation":
        return ParametrisedSimilarityMetric(similarity_metric.gradient_correlation, gradient_method=gradient_method)
    raise ValueError(f"Unknown similarity metric '{config_string}'.")


def run_experiment(*, reg_config: RegConfig, exp_config: ExperimentConfig, device: torch.device,
                   tqdm_position: int = 0) -> torch.Tensor | None:
    """

    :param reg_config:
    :param exp_config:
    :param device:
    :param tqdm_position:
    :return: A tensor of size (iteration count,) or None; the distance from g.t. of the optimisation at each
    iteration, averaged over `sample_count_per_distance` repetitions, unless the configuration is trivial /
    unnecessary, in which case `None`.
    """
    data_manager().set_data("ct_path", exp_config.ct_path, check_equality=True)
    data_manager().set_data("downsample_level", exp_config.downsample_level, check_equality=True)
    data_manager().set_data("truncation_percent", exp_config.truncation_percent, check_equality=True)
    # -----
    # Configuring according to desired cropping technique
    if exp_config.cropping == "None":
        apply_cropping = None
        data_manager().set_data("cropping", None, check_equality=True)
    elif exp_config.cropping == "nonzero_drr":
        apply_cropping = args_from_dag()(geometry.get_crop_nonzero_drr)
    elif exp_config.cropping == "full_depth_drr":
        apply_cropping = args_from_dag()(geometry.get_crop_full_depth_drr)
    else:
        raise ValueError(f"Unknown cropping technique '{exp_config.cropping}'.")
    # -----
    # Configuring according to desired similarity metric
    p_sim_met: ParametrisedSimilarityMetric = string_to_sim_met(exp_config.sim_metric)
    # -----
    # Configuring according to desired masking technique
    if exp_config.mask == "None":
        apply_mask = False
        data_manager().set_data("mask_transformation", None, check_equality=True)
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
    # Defining the objective function accordingly
    def objective_function(parameters: torch.Tensor) -> torch.Tensor:
        t = mapping_parameters_to_transformation(parameters)
        # Setting the parameters
        data_manager().set_data("current_transformation", t)
        if apply_cropping is not None:
            data_manager().set_data("cropping", apply_cropping())
        if apply_mask:
            data_manager().set_data("mask_transformation", t)
        # Getting the resulting moving and fixed images
        moving_image = data_manager().get("moving_image")
        fixed_image = data_manager().get("fixed_image")
        # Comparing, potentially weighting with a mask
        if apply_mask:
            mask = data_manager().get("mask")
            if weight_with_mask:
                return -p_sim_met.func_weighted(moving_image, fixed_image, mask)
        return -p_sim_met.func(moving_image, fixed_image)

    # -----
    # Running repeated registrations with configured parameters
    dimensionality = 6
    distance_samples = torch.empty([int(exp_config.sample_count_per_distance), int(reg_config.iteration_count)],
                                   dtype=torch.float32, device=device)  # size = (sample count, iteration count)
    ground_truth = mapping_transformation_to_parameters(data_manager().get("transformation_gt"))
    for i in tqdm(  #
            range(int(exp_config.sample_count_per_distance)),  #
            desc="Repeated samples",  #
            position=tqdm_position,  #
            leave=None  #
    ):
        starting_params = random_parameters_at_distance(
            mapping_transformation_to_parameters(data_manager().get("transformation_gt")), exp_config.starting_distance)
        res = run_reg(  #
            obj_fun=objective_function,  #
            config=reg_config,  #
            starting_params=starting_params,  #
            device=device,  #
            tqdm_position=tqdm_position + 1)  # size = (iteration count, dimensionality + 1)
        distance_samples[i, :] = torch.linalg.vector_norm(res[:, 0:dimensionality] - ground_truth,
                                                          dim=1)  # size = (iteration count,)

    return distance_samples.mean(dim=0)  # size = (iteration count,)


def run_experiments(*, params_to_vary: dict[str, list | torch.Tensor], constants: dict[str, Any],
                    output_directory: pathlib.Path, device: torch.device, tqdm_position: int = 0) -> None:
    assert output_directory.is_dir()
    each_range_length = []
    for name, values in params_to_vary.items():
        if isinstance(values, torch.Tensor):
            assert len(values.size()) == 1
        each_range_length.append(len(values))
    total = 1
    for l in each_range_length:
        total *= l
    logger.info(f"Running experiments with the following constant parameters:\n{pprint.pformat(constants)}")
    tqdm_iterator = tqdm(itertools.product(*(range(l) for l in each_range_length)), desc="Experiments", total=total,
                         position=tqdm_position, leave=None)
    for indices in tqdm_iterator:
        instance_specific = {  #
            name: values[index]  #
            for index, (name, values) in zip(indices, params_to_vary.items())  #
        }  # config specific to this instance
        tqdm_iterator.set_postfix(**instance_specific)  # displaying this
        instance_all = instance_specific | constants  # all the config for this instance
        # Separating the config into registration and experiment configs
        exp_config_by_name = copy.deepcopy(instance_all)
        try:
            reg_config = RegConfig(  #
                particle_count=exp_config_by_name.pop("particle_count"),  #
                particle_initialisation_spread=exp_config_by_name.pop("particle_initialisation_spread"),  #
                iteration_count=exp_config_by_name.pop("iteration_count"))
        except Exception as e:
            logger.error(f"Error constructing registration configuration at indices {indices}: {e}")
            continue
        try:
            exp_config = ExperimentConfig(**exp_config_by_name)
        except Exception as e:
            logger.error(f"Error constructing experiment configuration at indices {indices}: {e}\nParameters:\n"
                         f"{pprint.pformat(instance_all)}")
            continue
        # Running the experiment
        try:
            res = run_experiment(reg_config=reg_config, exp_config=exp_config, device=device,
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
        # Getting the rows for the DataFrame and saving
        df = pd.DataFrame([  #
            instance_all | {"iteration": iteration, "distance": res[iteration].item()}  #
            for iteration in range(len(res))  #
        ])
        df.to_parquet(output_directory / f"data_{"_".join([str(i) for i in indices])}.parquet")


def main(*, cache_directory: str, ct_path: str, data_output_dir: str | pathlib.Path, show: bool = False):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_data_manager()
    err = data_manager().set_data_multiple(  #
        device=device,  #
        ct_path=ct_path,  #
        cache_directory=cache_directory,  #
        save_to_cache=False,  #
        regenerate_drr=True,  #
        new_drr_size=torch.Size([1000, 1000]),  #
        truncation_percent=0,  #
        cropping=None,  #
        source_offset=torch.zeros(2),  #
        downsample_level=2,  #
        ap_transformation=Transformation(rotation=torch.tensor([0.5 * torch.pi, 0.0, 0.0], device=device),
                                         translation=torch.zeros(3, device=device)),  #
        target_ap_distance=5.0,  #
        current_transformation=Transformation.random_uniform(device=device),  #
        mask_transformation=None  #
    )
    if isinstance(err, Error):
        logger.error(f"Error setting initial data values: {err.description}")
        return
    err = data_manager().add_updater("load_untruncated_ct", load_untruncated_ct)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    err = data_manager().add_updater("set_target_image", set_target_image)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    err = data_manager().add_updater("apply_truncation", apply_truncation)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    err = data_manager().add_updater("refresh_image_2d_scale_factor", updaters.refresh_image_2d_scale_factor)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    err = data_manager().add_updater("refresh_hyperparameter_dependent", updaters.refresh_hyperparameter_dependent)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    err = data_manager().add_updater("refresh_mask_transformation_dependent",
                                     updaters.refresh_mask_transformation_dependent)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    err = data_manager().add_updater("project_drr", project_drr)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return

    data_manager().set_data("current_transformation", data_manager().get("transformation_gt"))

    data_manager().set_data("truncation_percent", 5)

    if show:
        plt.ion()  # figures are non-blocking
        plt.show()
        fig, axes = plt.subplots(1, 4)
        image_2d_full = data_manager().get("image_2d_full")
        axes[0].imshow(image_2d_full.cpu().numpy())
        axes[0].set_title("original target")
        fixed_image = data_manager().get("fixed_image")
        if isinstance(fixed_image, Error):
            raise RuntimeError(f"Error getting fixed image: {fixed_image.description}")
        axes[1].imshow(fixed_image.cpu().numpy())
        axes[1].set_title("fixed image")
        moving_image = data_manager().get("moving_image")
        axes[2].imshow(moving_image.cpu().numpy())
        axes[2].set_title("moving image at G.T.")
        data_manager().set_data("mask_transformation", data_manager().get("current_transformation"))
        mask = data_manager().get("mask")
        axes[3].imshow(mask.cpu().numpy())
        axes[3].set_title("mask at G.T.")
        logger.info(f"ZNCC at G.T. with masking = "
                    f"{-similarity_metric.weighted_local_ncc(moving_image, fixed_image, mask, kernel_size=8)}")
        plt.draw()
        plt.pause(0.1)

    constants = {  #
        # ExperimentConfig
        "ct_path": data_manager().get("ct_path"),  #
        "downsample_level": 1,  #
        # - varying "truncation_percent": 15,  #
        # - varying - "cropping": "None",  #
        # - varying - "mask": "None",  #
        "sim_metric": "zncc",  #
        "starting_distance": 10.0,  #
        "sample_count_per_distance": 10,  #
        # RegConfig
        "particle_count": 2000,  #
        "particle_initialisation_spread": 5.0,  #
        "iteration_count": 10,  #
    }

    if show:
        starting_params = random_parameters_at_distance(
            mapping_transformation_to_parameters(data_manager().get("transformation_gt")), 15.0)
        # data_manager().set_data("current_transformation", mapping_parameters_to_transformation(starting_params))
        # set_crop_to_nonzero_drr()
        data_manager().set_data("truncation_percent", 30)

        def objective_function(parameters: torch.Tensor) -> torch.Tensor:
            data_manager().set_data("current_transformation", mapping_parameters_to_transformation(parameters))
            _moving_image = data_manager().get("moving_image")
            _fixed_image = data_manager().get("fixed_image")
            return -similarity_metric.ncc(_moving_image, _fixed_image)

        res = run_reg(  #
            obj_fun=lambda p: objective_function,  #
            starting_params=starting_params, config=RegConfig(  #
                particle_count=constants["particle_count"],  #
                particle_initialisation_spread=constants["particle_initialisation_spread"],  #
                iteration_count=constants["iteration_count"],  #
            ),  #
            device=device,  #
            plot="mask")  # size = (iteration count, dimensionality + 1)
        logger.info(f"Result: {res}")
        plt.ioff()  # figures are blocking
        plt.show()  # return
        return

    instance_output_dir: pathlib.Path = instance_output_directory(data_output_dir)

    (instance_output_dir / "notes.txt").write_text(  #
        "Varying truncation and cropping and masking; finished off the previous dataset: 2026-02-03_16-49-22; just "
        "the last two dataframes.")

    run_experiments(params_to_vary={  #
        "truncation_percent": [45],  #
        "cropping": ["full_depth_drr"],  #
        "mask": ["Every evaluation",  #
                 "Every evaluation weighting zncc"],  #
    }, output_directory=instance_output_dir, constants=constants, device=device)


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
    # parser.add_argument("-i", "--no-load", action='store_true',
    #                     help="Do not load any pre-calculated data from the cache.")
    # parser.add_argument(
    #     "-r", "--regenerate-drr", action='store_true',
    #     help="Regenerate the DRR through the 3D data, regardless of whether a DRR has been cached.")
    # parser.add_argument("-n", "--no-save", action='store_true', help="Do not save any data to the cache.")
    parser.add_argument("-n", "--notify", action="store_true", help="Send notification on completion.")
    parser.add_argument("-s", "--show", action="store_true", help="Show images at the G.T. alignment.")
    parser.add_argument("-d", "--data-output-dir", type=str, default="data/temp/program_truncation",
                        help="Directory in which to save output data.")
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    try:
        main(cache_directory=args.cache_directory, ct_path=args.ct_path if "ct_path" in vars(args) else None,
             data_output_dir=args.data_output_dir, show=args.show)
        if args.notify:
            pushover.send_notification(__file__, "Script finished.")
    except Exception as e:
        if args.notify:
            pushover.send_notification(__file__, "Script raised exception: {}.".format(e))
        raise e
