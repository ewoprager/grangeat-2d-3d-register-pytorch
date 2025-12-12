import torch
from typing import Any

import napari

from notification import logs_setup
from program.lib.structs import Error
from program import init_data_manager, data_manager, dag_updater
from program.modules.interface import init_viewer, viewer, FixedImageGUI, MovingImageGUI, RegisterGUI
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


def of(x: Transformation) -> torch.Tensor:
    return x.vectorised().sum()


def main():
    init_data_manager()
    init_viewer(title="Program Test")
    fixed_image_gui = FixedImageGUI()
    moving_image_gui = MovingImageGUI()
    register_gui = RegisterGUI({"the_only_one": of})
    data_manager().add_updater("fixed_image_updater", fixed_image_updater)
    data_manager().add_updater("project_drr", project_drr)
    data_manager().add_updater("load_ct", updaters.load_ct)
    data_manager().add_updater("set_synthetic_target_image", updaters.set_synthetic_target_image)
    data_manager().set_data_multiple({  #
        "fixed_image_size": torch.Size([500, 500]),  #
        "regenerate_drr": True,  #
        "cache_directory": "cache",  #
        "save_to_cache": True,  #
        "new_drr_size": torch.Size([500, 500]),  #
        "ct_path": "/Users/eprager/Library/CloudStorage/OneDrive-UniversityofCambridge/CUED/4th Year Project/Data/First/ct",
        #
        "device": torch.device("cpu"),  #
        "current_transformation": Transformation.zero()  #
    })
    value = data_manager().get("moving_image")
    if isinstance(value, Error):
        logger.error(f"Couldn't get moving image: {value.description}.")
        return

    napari.run()


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    main()
