import argparse
import os
from typing import Any

import torch
import napari

from reg23_experiments.notification import logs_setup, pushover
from reg23_experiments.program.lib.structs import Error
from reg23_experiments.program import init_data_manager, data_manager, dag_updater
from reg23_experiments.program.modules.interface import init_viewer, viewer, FixedImageGUI, MovingImageGUI, RegisterGUI
from reg23_experiments.registration.lib.structs import Transformation, SceneGeometry
from reg23_experiments.registration.lib.geometry import generate_drr
from reg23_experiments.program import updaters, args_from_dag
from reg23_experiments.registration.objective_function import ncc


@dag_updater(names_returned=["fixed_image"])
def fixed_image_updater(images_2d_full: list[torch.Tensor]) -> dict[str, Any]:
    return {"fixed_image": images_2d_full[0]}


@dag_updater(names_returned=["moving_image"])
def project_drr(ct_volumes: list[torch.Tensor], ct_spacing: torch.Tensor, current_transformation: Transformation,
                fixed_image_size: torch.Size, source_distance: float, fixed_image_spacing: torch.Tensor) -> dict[
    str, Any]:
    return {"moving_image": generate_drr(ct_volumes[0], transformation=current_transformation, voxel_spacing=ct_spacing,
                                         detector_spacing=fixed_image_spacing,
                                         scene_geometry=SceneGeometry(source_distance=source_distance),
                                         output_size=fixed_image_size)}


@args_from_dag(names_left=["transformation"])
def of(transformation: Transformation, ct_volumes: list[torch.Tensor], ct_spacing: torch.Tensor,
       fixed_image_size: torch.Size, source_distance: float, fixed_image_spacing: torch.Tensor,
       fixed_image: torch.Tensor) -> torch.Tensor:
    moving_image = generate_drr(ct_volumes[0], transformation=transformation, voxel_spacing=ct_spacing,
                                detector_spacing=fixed_image_spacing,
                                scene_geometry=SceneGeometry(source_distance=source_distance),
                                output_size=fixed_image_size)
    return ncc(moving_image, fixed_image)


def main(*, ct_path: str, cache_directory: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "mps") if torch.mps.is_available() else torch.device("cpu")

    init_data_manager()
    init_viewer(title="Program Test")
    fixed_image_gui = FixedImageGUI()
    moving_image_gui = MovingImageGUI()
    register_gui = RegisterGUI({"the_only_one": of})
    data_manager().add_updater("fixed_image_updater", fixed_image_updater)
    data_manager().add_updater("project_drr", project_drr)
    data_manager().add_updater("load_ct", updaters.load_ct)
    data_manager().add_updater("set_synthetic_target_image", updaters.set_synthetic_target_image)
    data_manager().set_data_multiple(  #
        fixed_image_size=torch.Size([500, 500]),  #
        regenerate_drr=True,  #
        cache_directory=cache_directory,  #
        save_to_cache=True,  #
        new_drr_size=torch.Size([500, 500]),  #
        ct_path=ct_path,  #
        device=device,  #
        current_transformation=Transformation.zero(device=device)  #
    )
    value = data_manager().get("moving_image")
    if isinstance(value, Error):
        logger.error(f"Couldn't get moving image: {value.description}.")
        return

    napari.run()


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
    # parser.add_argument("-s", "--show", action="store_true", help="Show images at the G.T. alignment.")
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    try:
        main(ct_path=args.ct_path, cache_directory=args.cache_directory)
        if args.notify:
            pushover.send_notification(__file__, "Script finished.")
    except Exception as e:
        if args.notify:
            pushover.send_notification(__file__, "Script raised exception: {}.".format(e))
        raise e
