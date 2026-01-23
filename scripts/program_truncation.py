import argparse
import os
from typing import Any, Callable, Literal, Sequence
from datetime import datetime
import types
import pprint

import pathlib
import torch
import torchviz
import matplotlib.pyplot as plt
import traitlets

from reg23_experiments.notification.console_logging import tqdm
from reg23_experiments.notification import logs_setup, pushover
from reg23_experiments.program import data_manager, init_data_manager, dag_updater, updaters, args_from_dag
from reg23_experiments.program.lib.structs import Error, StrictHasTraits
from reg23_experiments.registration import data, drr
from reg23_experiments.registration.lib.optimisation import mapping_transformation_to_parameters, \
    mapping_parameters_to_transformation, random_parameters_at_distance
from reg23_experiments.registration.lib.structs import Transformation, SceneGeometry
from reg23_experiments.registration.lib import geometry
from reg23_experiments.registration import objective_function as lib_objective_function
from reg23_experiments.registration.interface.lib.structs import HyperParameters, Cropping
from reg23_experiments.pso import swarm as pso


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
    ct_volume, ct_spacing = data.load_volume(pathlib.Path(ct_path))
    ct_volume = ct_volume.to(device=device, dtype=torch.float32)
    ct_spacing = ct_spacing.to(device=device)

    if ct_permutation is not None:
        assert len(ct_permutation) == 3
        ct_volume = ct_volume.permute(*ct_permutation)
        ct_spacing = ct_spacing[torch.tensor(ct_permutation)]

    return {"untruncated_ct_volume": ct_volume, "ct_spacing": ct_spacing}


@dag_updater(names_returned=["source_distance", "images_2d_full", "fixed_image_spacing", "transformation_gt"])
def set_target_image(ct_path: str, ct_spacing: torch.Tensor, untruncated_ct_volume: torch.Tensor,
                     new_drr_size: torch.Size, regenerate_drr: bool, save_to_cache: bool, cache_directory: str,
                     ap_transformation: Transformation, target_ap_distance: float) -> dict[str, Any]:
    # generate a DRR through the volume
    drr_spec = None
    if not regenerate_drr and ct_path is not None:
        drr_spec = data.load_cached_drr(cache_directory, ct_path)

    if drr_spec is None:
        tr = mapping_parameters_to_transformation(
            random_parameters_at_distance(mapping_transformation_to_parameters(ap_transformation), target_ap_distance))
        drr_spec = drr.generate_drr_as_target(cache_directory, ct_path, untruncated_ct_volume, ct_spacing,
                                              save_to_cache=save_to_cache, size=new_drr_size, transformation=tr)

    fixed_image_spacing, scene_geometry, image_2d_full, transformation_ground_truth = drr_spec
    del drr_spec

    # Generating X-ray mipmap
    down_sampler = torch.nn.AvgPool2d(2)
    images_2d_full = [image_2d_full]
    while min(images_2d_full[-1].size()) > 1:
        images_2d_full.append(down_sampler(images_2d_full[-1].unsqueeze(0))[0])

    return {"source_distance": scene_geometry.source_distance, "images_2d_full": images_2d_full,
            "fixed_image_spacing": fixed_image_spacing, "transformation_gt": transformation_ground_truth}


@dag_updater(names_returned=["ct_volumes"])
def apply_truncation(untruncated_ct_volume: torch.Tensor, truncation_fraction: float) -> dict[str, Any]:
    # truncate the volume
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
                hyperparameters: HyperParameters, translation_offset: torch.Tensor, fixed_image_offset: torch.Tensor,
                image_2d_downsample_level: int, device) -> dict[str, Any]:
    # Applying the translation offset
    new_translation = current_transformation.translation + torch.cat(
        (torch.tensor([0.0], device=device, dtype=current_transformation.translation.dtype),
         translation_offset.to(device=current_transformation.device)))
    transformation = Transformation(rotation=current_transformation.rotation, translation=new_translation).to(
        device=device)

    return {"moving_image": geometry.generate_drr(ct_volumes[hyperparameters.downsample_level],
                                                  transformation=transformation,
                                                  voxel_spacing=ct_spacing * 2.0 ** hyperparameters.downsample_level,
                                                  detector_spacing=fixed_image_spacing * 2.0 ** image_2d_downsample_level,
                                                  scene_geometry=SceneGeometry(source_distance=source_distance,
                                                                               fixed_image_offset=fixed_image_offset),
                                                  output_size=fixed_image_size)}


@dag_updater(names_returned=["current_loss"])
def of_zncc(moving_image: torch.Tensor, fixed_image: torch.Tensor) -> dict[str, Any]:
    return {"current_loss": -lib_objective_function.ncc(moving_image, fixed_image)}


def reset_crop(images_2d_full: list[torch.Tensor]) -> None:
    size = images_2d_full[0].size()
    logger.info(f"Resetting crop for new value of 'images_2d_full'; size = [{size[0]} x {size[1]}].")
    res = data_manager().get("hyperparameters")
    new_value = HyperParameters(  #
        cropping=Cropping.zero(size),  #
        source_offset=torch.zeros(2),  #
        downsample_level=2  #
    ) if isinstance(res, Error) else HyperParameters(  #
        cropping=Cropping.zero(size),  #
        source_offset=res.source_offset,  #
        downsample_level=res.downsample_level  #
    )
    err = data_manager().set_data("hyperparameters", new_value)
    if isinstance(err, Error):
        logger.error(f"Error setting hyperparameters with reset cropping: {err.description}")


def distance_distribution_delta(nominal_distance: float) -> float:
    return nominal_distance


def obj_func(parameters: torch.Tensor) -> torch.Tensor:
    data_manager().set_data("current_transformation", mapping_parameters_to_transformation(parameters))
    return data_manager().get("current_loss")


def obj_func_masked(parameters: torch.Tensor) -> torch.Tensor:
    t = mapping_parameters_to_transformation(parameters)
    data_manager().set_data("current_transformation", t)
    data_manager().set_data("mask_transformation", t)
    return data_manager().get("current_loss")


class RegConfig(StrictHasTraits):
    particle_count = traitlets.Int(default_value=traitlets.Undefined)
    particle_initialisation_spread = traitlets.Float(default_value=traitlets.Undefined)
    iteration_count = traitlets.Int(default_value=traitlets.Undefined)


def run_reg(*, objective_function: Callable, starting_params: torch.Tensor, config: RegConfig, device: torch.device,
            plot: Literal["no", "yes", "mask"] = "no", tqdm_position: int = 0) -> torch.Tensor:
    """

    :param objective_function:
    :param starting_params:
    :param config:
    :param device:
    :param plot:
    :param tqdm_position:
    :return: A tensor of size (iteration count, dimensionality + 1), where each row corresponds to an iteration of the optimisation, and stores the following data: | <- position of current best -> | current best |
    """
    if plot != "no":
        ncols = 2
        if plot == "mask":
            ncols += 1
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

    pso_config = pso.OptimisationConfig(objective_function=objective_function)
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
                axes[2].set_title("fixed image")
                axes[2].imshow(data_manager().get("fixed_image").cpu().numpy())
            plt.draw()
            plt.pause(0.1)

        tqdm_iterator.update()
        tqdm_iterator.set_postfix(best=swarm.current_optimum.item())
    return ret


class ExperimentConfig(StrictHasTraits):
    distance_distribution = traitlets.Callable(default_value=traitlets.Undefined)
    nominal_distances = traitlets.Instance(torch.Tensor, default_value=traitlets.Undefined)
    sample_count_per_distance = traitlets.Int(default_value=traitlets.Undefined)


def run_experiment(*, reg_config: RegConfig, exp_config: ExperimentConfig, objective_function: Callable,
                   device: torch.device, tqdm_position: int = 0) -> torch.Tensor:
    """

    :param reg_config:
    :param exp_config:
    :param objective_function:
    :param device:
    :param tqdm_position:
    :return: A tensor of size (nominal distance count, iteration count); the distance from g.t. of the optimisation at each iteration, averaged over `sample_count_per_distance` repetitions, for each nominal starting distance
    """
    dimensionality = 6
    ret = torch.empty([exp_config.nominal_distances.numel(), reg_config.iteration_count], dtype=torch.float32,
                      device=device)  # size = (nominal distance count, iteration count)
    tqdm_nominal_distances = tqdm(  #
        range(exp_config.nominal_distances.numel()),  #
        desc="Nominal distances",  #
        position=tqdm_position,  #
        leave=None)
    ground_truth = mapping_transformation_to_parameters(data_manager().get("transformation_gt"))
    for j in tqdm_nominal_distances:
        tqdm_nominal_distances.set_postfix_str("Nominal distance {:.3f}".format(exp_config.nominal_distances[j].item()))
        distance_generator = lambda: exp_config.distance_distribution(exp_config.nominal_distances[j].item())
        distance_samples = torch.empty([int(exp_config.sample_count_per_distance), reg_config.iteration_count],
                                       dtype=torch.float32, device=device)  # size = (sample count, iteration count)
        for i in tqdm(  #
                range(int(exp_config.sample_count_per_distance)),  #
                desc="Repeated samples",  #
                position=tqdm_position + 1,  #
                leave=None  #
        ):
            starting_params = random_parameters_at_distance(
                mapping_transformation_to_parameters(data_manager().get("transformation_gt")), distance_generator())
            res = run_reg(  #
                objective_function=objective_function,  #
                config=reg_config,  #
                starting_params=starting_params,  #
                device=device,  #
                tqdm_position=tqdm_position + 2)  # size = (iteration count, dimensionality + 1)
            distance_samples[i] = torch.linalg.vector_norm(res[:, 0:dimensionality] - ground_truth,
                                                           dim=1)  # size = (iteration count,)
        ret[j] = distance_samples.mean(dim=0)  # size = (iteration count,)

    return ret


def main(*, cache_directory: str, ct_path: str | None, data_output_dir: str | pathlib.Path, show: bool = False):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance_output_dir: pathlib.Path = instance_output_directory(data_output_dir)

    init_data_manager()
    err = data_manager().set_data_multiple(  #
        device=device,  #
        ct_path=ct_path,  #
        cache_directory=cache_directory,  #
        save_to_cache=False,  #
        regenerate_drr=True,  #
        new_drr_size=torch.Size([1000, 1000]),  #
        truncation_fraction=0.0,  #
        hyperparameters=HyperParameters(  #
            cropping=None,  #
            source_offset=torch.zeros(2),  #
            downsample_level=2  #
        ),  #
        ap_transformation=Transformation(rotation=torch.tensor([0.5 * torch.pi, 0.0, 0.0], device=device),
                                         translation=torch.zeros(3, device=device)),  #
        target_ap_distance=5.0,  #
        current_transformation=Transformation.random_uniform(device=device),  #
        mask_transformation=Transformation.zero(device=device)  #
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
    err = data_manager().add_updater("refresh_image_2d_downsample_level", updaters.refresh_image_2d_downsample_level)
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
    err = data_manager().add_updater("sim_metric", of_zncc)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return

    data_manager().add_callback("images_2d_full", "reset_crop_on_img2dfull", reset_crop)

    data_manager().set_data("current_transformation", data_manager().get("transformation_gt"))

    data_manager().set_data("truncation_fraction", 0.05)

    if show:
        plt.ion()  # figures are non-blocking
        plt.show()
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(data_manager().get("fixed_image").cpu().numpy())
        axes[0].set_title("fixed image")
        axes[1].imshow(data_manager().get("moving_image").cpu().numpy())
        axes[1].set_title("moving image at G.T.")
        logger.info(f"ZNCC at G.T. = {data_manager().get("current_loss")}")
        plt.draw()
        plt.pause(0.1)

    truncation_fraction_count = 1
    nominal_distance_count = 3
    sample_count_per_distance = 10
    iteration_count = 15

    reg_config = RegConfig(particle_count=1000, particle_initialisation_spread=5.0, iteration_count=iteration_count)

    if False:
        nominal_distance = 5.0
        distance_distribution = distance_distribution_delta
        distance_generator = lambda: distance_distribution(nominal_distance)
        starting_params = random_parameters_at_distance(
            mapping_transformation_to_parameters(data_manager().get("transformation_gt")), distance_generator())
        res = run_reg(objective_function=obj_func, starting_params=starting_params, config=reg_config, device=device,
                      plot="mask")  # size = (iteration count, dimensionality + 1)
        logger.info(f"Result: {res}")
        plt.ioff()  # figures are blocking
        plt.show()  # return
        return

    exp_config = ExperimentConfig(distance_distribution=distance_distribution_delta,
                                  nominal_distances=torch.linspace(0.1, 20.0, nominal_distance_count),
                                  sample_count_per_distance=sample_count_per_distance)

    truncation_fractions = torch.linspace(0.0, 0.6, truncation_fraction_count)

    # config
    save_dict(  #
        {  #
            "ct_path": data_manager().get("ct_path"),  #
            "x_ray_path": "DRR",  #
            "notes": "Just varying truncation fraction",  #
        } | configs_to_dict(reg_config, exp_config),  #
        directory=instance_output_dir,  #
        stem="config")

    # shared parameters
    save_dict({  #
        "cropping": "None",  #
        "sim_metric": "zncc",  #
        "downsample_level": data_manager().get("hyperparameters").downsample_level,  #
    }, directory=instance_output_dir, stem="shared_parameters")

    import sys
    logger.info(sys.stderr.isatty())

    logger.info("##### No masking #####")
    tqdm_truncation_fraction = tqdm(truncation_fractions, desc="Truncation fractions")
    for truncation_fraction in tqdm_truncation_fraction:
        tqdm_truncation_fraction.set_postfix_str("Truncation fraction {:.3f}".format(truncation_fraction.item()))
        data_manager().set_data("truncation_fraction", float(truncation_fraction))
        results = run_experiment(  #
            reg_config=reg_config,  #
            exp_config=exp_config,  #
            objective_function=obj_func,  #
            device=device,  #
            tqdm_position=1)  # size = (nominal distance count, iteration count)
        results_dir: pathlib.Path = instance_output_dir / "truncation_fraction_{:.3f}".format(
            float(truncation_fraction)).replace(".", "p")
        results_dir.mkdir(exist_ok=True)
        torch.save(results, results_dir / "convergence_series.pkl")
        # parameters
        save_dict({  #
            "truncation_fraction": truncation_fraction.item(),  #
            "mask": "None",  #
        }, directory=results_dir, stem="parameters")

    logger.info("##### Yes masking #####")
    tqdm_truncation_fraction = tqdm(truncation_fractions, desc="Truncation fractions")
    for truncation_fraction in tqdm_truncation_fraction:
        tqdm_truncation_fraction.set_postfix_str("Truncation fraction {:.3f}".format(truncation_fraction.item()))
        data_manager().set_data("truncation_fraction", float(truncation_fraction))
        results = run_experiment(  #
            reg_config=reg_config,  #
            exp_config=exp_config,  #
            objective_function=obj_func_masked,  #
            device=device,  #
            tqdm_position=1)  # size = (nominal distance count, iteration count)
        results_dir: pathlib.Path = instance_output_dir / "masked_truncation_fraction_{:.3f}".format(
            float(truncation_fraction)).replace(".", "p")
        results_dir.mkdir(exist_ok=True)
        torch.save(results, results_dir / "convergence_series.pkl")
        # parameters
        save_dict({  #
            "truncation_fraction": truncation_fraction.item(),  #
            "mask": "Every evaluation",  #
        }, directory=results_dir, stem="parameters")


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-c", "--cache-directory", type=str, default="cache",
                        help="Set the directory where data that is expensive to calculate will be saved. The default "
                             "is 'cache'.")
    parser.add_argument("-p", "--ct-path", type=str,
                        help="Give a path to a .nrrd file, .nii file or directory of .dcm files containing CT data to process. If not "
                             "provided, some simple synthetic data will be used instead - note that in this case, data will not be "
                             "saved to the cache.")
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
