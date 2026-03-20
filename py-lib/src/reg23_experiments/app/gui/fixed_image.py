import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.app.gui.viewer_singleton import viewer
from reg23_experiments.app.state import AppState

__all__ = ["FixedImageGUI", "Image2DFullGUI"]


class FixedImageGUI:
    def __init__(self, app_state: AppState, namespace: str | None = None):
        self._app_state = app_state
        fixed_image_key = "fixed_image" if namespace is None else f"{namespace}__fixed_image"
        self._app_state.dadg.set_evaluation_laziness(fixed_image_key, lazily_evaluated=False)
        value = self._app_state.dadg.get(fixed_image_key, soft=True)
        if isinstance(value, Error):
            raise RuntimeError(f"Error softly getting '{fixed_image_key}' from DADG: {value.description}.")
        initial_image = value if isinstance(value, torch.Tensor) else torch.zeros((2, 2))
        self._layer = viewer().add_image(initial_image.cpu().numpy(), colormap="yellow", interpolation2d="linear",
                                         name=f"Fixed image {namespace}")
        self._app_state.dadg.observe(fixed_image_key, "interface", self._observer_callback)

    def _observer_callback(self, new_value: torch.Tensor) -> None:
        self._layer.data = new_value.cpu().numpy()


class Image2DFullGUI:
    def __init__(self, app_state: AppState, namespace: str | None = None):
        self._app_state = app_state
        image_2d_full_key = "image_2d_full" if namespace is None else f"{namespace}__image_2d_full"
        self._app_state.dadg.set_evaluation_laziness(image_2d_full_key, lazily_evaluated=False)
        value = self._app_state.dadg.get(image_2d_full_key, soft=True)
        if isinstance(value, Error):
            raise RuntimeError(f"Error softly getting '{image_2d_full_key}' from DADG: {value.description}.")
        initial_image = value if isinstance(value, torch.Tensor) else torch.zeros((2, 2))
        self._layer = viewer().add_image(initial_image.cpu().numpy(), colormap="yellow", interpolation2d="linear",
                                         name=f"Image 2D Full {namespace}")
        self._app_state.dadg.observe(image_2d_full_key, "interface", self._observer_callback)

    def _observer_callback(self, new_value: torch.Tensor) -> None:
        self._layer.data = new_value.cpu().numpy()
