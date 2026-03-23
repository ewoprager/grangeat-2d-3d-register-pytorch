import napari.layers
import torch
import weakref

from reg23_experiments.data.structs import Error
from reg23_experiments.app.gui.viewer_singleton import viewer
from reg23_experiments.app.state import AppState

__all__ = ["add_image_layer"]


class _ImageLayerManager:
    def __init__(self, *, layer: napari.layers.Layer, app_state: AppState, dadg_key: str):
        self._layer = weakref.ref(layer)
        self._app_state = app_state
        self._dadg_key = dadg_key
        self._app_state.dadg.set_evaluation_laziness(self._dadg_key, lazily_evaluated=False)
        self._app_state.dadg.observe(self._dadg_key, "image_manager", self._observer_callback)

    def __del__(self):
        self._app_state.dadg.set_evaluation_laziness(self._dadg_key, lazily_evaluated=True)

    def _observer_callback(self, new_value: torch.Tensor) -> None:
        self._layer().data = new_value.cpu().numpy()


def add_image_layer(*, app_state: AppState, dadg_key: str, colormap: str = "yellow") -> napari.layers.Layer:
    value = app_state.dadg.get(dadg_key, soft=True)
    if isinstance(value, Error):
        raise RuntimeError(f"Error softly getting '{dadg_key}' from DADG: {value.description}.")
    initial_image = value if isinstance(value, torch.Tensor) else torch.zeros((500, 500))
    layer = viewer().add_image(initial_image.cpu().numpy(), colormap=colormap, blending="additive",
                               interpolation2d="linear", name=dadg_key)
    layer.my_plugin = _ImageLayerManager(layer=layer, app_state=app_state, dadg_key=dadg_key)
    return layer
