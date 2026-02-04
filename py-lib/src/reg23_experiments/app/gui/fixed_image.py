import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.app.gui.viewer_singleton import viewer
from reg23_experiments.app.state import AppState

__all__ = ["FixedImageGUI"]


class FixedImageGUI:
    def __init__(self, app_state: AppState):
        self._app_state = app_state
        self._app_state.dag.set_evaluation_laziness("fixed_image", lazily_evaluated=False)
        value = self._app_state.dag.get("fixed_image", soft=True)
        if isinstance(value, Error):
            raise RuntimeError(f"Error softly getting 'fixed_image' from DAG: {value.description}.")
        initial_image = value if isinstance(value, torch.Tensor) else torch.zeros((2, 2))
        self._layer = viewer().add_image(initial_image.cpu().numpy(), colormap="yellow", interpolation2d="linear",
                                         name="Fixed image")
        self._app_state.dag.add_callback("fixed_image", "interface", self._set_callback)

    def _set_callback(self, new_value: torch.Tensor) -> None:
        self._layer.data = new_value.cpu().numpy()
