import torch
from typing import Any

import napari

from notification import logs_setup
from program.lib.structs import Error
from program import init_data_manager, data_manager, dag_updater
from program.modules.interface import init_viewer, viewer, FixedImage, MovingImage
from registration.lib.structs import Transformation, SceneGeometry
from registration.lib.geometry import generate_drr
from program import updaters


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

def main():
    init_data_manager(lazy=False)
    init_viewer(title="Program Test")
    fixed_image = FixedImage()
    moving_image = MovingImage()
    data_manager().add_updater("fixed_image_updater", fixed_image_updater)
    data_manager().add_updater("project_drr", project_drr)
    data_manager().add_updater("load_ct", updaters.load_ct)
    data_manager().add_updater("set_synthetic_target_image", updaters.set_synthetic_target_image)
    data_manager().set_data("fixed_image_size", torch.Size([500, 500]))
    data_manager().set_data("regenerate_drr", True)
    data_manager().set_data("cache_directory", "cache")
    data_manager().set_data("save_to_cache", True)
    data_manager().set_data("new_drr_size", torch.Size([500, 500]))
    data_manager().set_data("ct_path",
                            "/Users/eprager/Library/CloudStorage/OneDrive-UniversityofCambridge/CUED/4th Year Project/Data/First/ct")
    data_manager().set_data("device", torch.device("cpu"))
    data_manager().set_data("current_transformation", Transformation.zero())
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}.")
    value = data_manager().get("moving_image")
    if isinstance(value, Error):
        logger.error(f"Couldn't get moving image: {value.description}.")
        return

    napari.run()


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    main()
