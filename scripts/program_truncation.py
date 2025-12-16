import argparse
import os
from typing import Any, Callable
import copy

import pathlib
import torch
import torchviz
import matplotlib.pyplot as plt
from tqdm import tqdm

from notification import logs_setup, pushover
from program import data_manager, init_data_manager, dag_updater, updaters, args_from_dag
from program.lib.structs import Error
from registration import data, drr
from registration.lib.optimisation import mapping_transformation_to_parameters
from registration.lib.structs import Transformation, SceneGeometry
from registration.lib import geometry
from registration import objective_function
from registration.interface.lib.structs import HyperParameters, Cropping


@dag_updater(names_returned=["untruncated_ct_volume", "ct_spacing"])
def load_untruncated_ct(ct_path: str, device) -> dict[str, Any]:
    ct_volume, ct_spacing = data.load_volume(pathlib.Path(ct_path))
    ct_volume = ct_volume.to(device=device, dtype=torch.float32)
    ct_spacing = ct_spacing.to(device=device)

    return {"untruncated_ct_volume": ct_volume, "ct_spacing": ct_spacing}


@dag_updater(names_returned=["source_distance", "images_2d_full", "fixed_image_spacing", "transformation_gt"])
def set_target_image(ct_path: str, ct_spacing: torch.Tensor, untruncated_ct_volume: torch.Tensor,
                     new_drr_size: torch.Size, regenerate_drr: bool, save_to_cache: bool, cache_directory: str) -> dict[
    str, Any]:
    # generate a DRR through the volume
    drr_spec = None
    if not regenerate_drr and ct_path is not None:
        drr_spec = data.load_cached_drr(cache_directory, ct_path)

    if drr_spec is None:
        drr_spec = drr.generate_drr_as_target(cache_directory, ct_path, untruncated_ct_volume, ct_spacing,
                                              save_to_cache=save_to_cache, size=new_drr_size)

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
                device) -> dict[str, Any]:
    # Applying the translation offset
    new_translation = current_transformation.translation + torch.cat(
        (torch.tensor([0.0], device=device, dtype=current_transformation.translation.dtype),
         translation_offset.to(device=current_transformation.device)))
    transformation = Transformation(rotation=current_transformation.rotation, translation=new_translation).to(
        device=device)

    return {"moving_image": geometry.generate_drr(ct_volumes[hyperparameters.downsample_level],
                                                  transformation=transformation,
                                                  voxel_spacing=ct_spacing * 2.0 ** hyperparameters.downsample_level,
                                                  detector_spacing=fixed_image_spacing * 2.0 ** hyperparameters.downsample_level,
                                                  scene_geometry=SceneGeometry(source_distance=source_distance,
                                                                               fixed_image_offset=fixed_image_offset),
                                                  output_size=fixed_image_size)}


@dag_updater(names_returned=["current_loss"])
def of_zncc(moving_image: torch.Tensor, fixed_image: torch.Tensor) -> dict[str, Any]:
    return {"current_loss": -objective_function.ncc(moving_image, fixed_image)}


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


def main(*, cache_directory: str, ct_path: str | None, show: bool = False):
    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_data_manager()
    err = data_manager().set_data_multiple(  #
        device=device,  #
        ct_path=ct_path,  #
        cache_directory=cache_directory,  #
        save_to_cache=True,  #
        regenerate_drr=True,  #
        new_drr_size=torch.Size([1000, 1000]),  #
        truncation_fraction=0.0,  #
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

    def get_grad() -> torch.Tensor:
        data_manager().get("current_transformation").rotation.requires_grad = True
        data_manager().get("current_transformation").translation.requires_grad = True
        loss = data_manager().get("current_loss")
        loss.backward()
        return mapping_transformation_to_parameters(
            Transformation(rotation=data_manager().get("current_transformation").rotation.grad,
                           translation=data_manager().get("current_transformation").translation.grad))

    data_manager().set_data("current_transformation", data_manager().get("transformation_gt"))

    if show:
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(data_manager().get("fixed_image").cpu().numpy())
        axes[1].imshow(data_manager().get("moving_image").cpu().numpy())
        plt.show()

    count = 200
    distances = torch.zeros(count)
    dots = torch.zeros(count)
    zero_vectorised = mapping_transformation_to_parameters(data_manager().get("transformation_gt"))
    for i in tqdm(range(count)):
        tr = Transformation.random_gaussian(  #
            rotation_mean=data_manager().get("transformation_gt").rotation,  #
            rotation_std=0.05,  #
            translation_mean=data_manager().get("transformation_gt").translation,  #
            translation_std=2.0  #
        )
        data_manager().set_data("current_transformation", tr)
        try:
            grad = get_grad()
        except RuntimeError as e:
            logger.warn(f"Runtime error in {e}")
            continue
        if (grad.abs() > 1e10).any():
            continue
        distances[i] = Transformation(rotation=tr.rotation.detach(), translation=tr.translation.detach()).distance(
            data_manager().get("transformation_gt"))
        dots[i] = torch.dot(torch.nn.functional.normalize(grad, dim=0),
                            torch.nn.functional.normalize(mapping_transformation_to_parameters(tr) - zero_vectorised,
                                                          dim=0))

    plt.scatter(distances.detach().cpu().numpy(), dots.detach().cpu().numpy())
    plt.show()


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
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    try:
        main(cache_directory=args.cache_directory, ct_path=args.ct_path if "ct_path" in vars(args) else None,
             show=args.show)
        if args.notify:
            pushover.send_notification(__file__, "Script finished.")
    except Exception as e:
        if args.notify:
            pushover.send_notification(__file__, "Script raised exception: {}.".format(e))
        raise e
