import torch

from program.lib.structs import Error
from program import data_manager
from program.modules.interface import viewer

__all__ = ["FixedImageGUI"]

class FixedImageGUI:
    def __init__(self):
        data_manager().add_callback("fixed_image", "interface", self._set_callback)
        data_manager().set_evaluation_laziness("fixed_image", lazily_evaluated=False)
        value = data_manager().get("fixed_image", soft=True)
        if isinstance(value, Error):
            raise RuntimeError(f"Error softly getting 'fixed_image' from DAG: {value.description}.")
        initial_image = value if isinstance(value, torch.Tensor) else torch.zeros((2, 2))
        self._layer = viewer().add_image(initial_image.cpu().numpy(), colormap="yellow", interpolation2d="linear",
                                         name="Fixed image")

    def _set_callback(self, new_value: torch.Tensor) -> None:
        self._layer.data = new_value.cpu().numpy()