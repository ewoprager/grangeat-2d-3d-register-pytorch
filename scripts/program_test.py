import torch
from typing import Any

import napari

from program.lib.structs import Error
from program import init_data_manager, data_manager, dag_updater
from program.modules.interface import init_viewer, viewer, FixedImage


@dag_updater(names_returned=["fixed_image"])
def fixed_image_updater(fixed_image_size: int) -> dict[str, Any]:
    return {"fixed_image": torch.rand((fixed_image_size, fixed_image_size))}


def main():
    init_data_manager(lazy=False)
    init_viewer(title="Program Test")
    fixed_image = FixedImage()
    data_manager().add_updater("fixed_image_updater", fixed_image_updater)
    data_manager().set_data("fixed_image_size", 100)

    napari.run()


if __name__ == "__main__":
    main()
