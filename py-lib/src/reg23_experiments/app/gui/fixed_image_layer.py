import logging

import napari.layers
import torch
import weakref

from reg23_experiments.data.structs import Error
from reg23_experiments.app.gui.viewer_singleton import viewer
from reg23_experiments.app.context import AppContext

__all__ = ["add_fixed_image_layer"]

logger = logging.getLogger(__name__)


class _FixedImageLayerManager:
    def __init__(self, *, ctx: AppContext, layer: napari.layers.Layer, dadg_key: str):
        self._ctx = ctx
        self._layer = weakref.ref(layer)
        self._dadg_key = dadg_key
        self._ctx.dadg.set_evaluation_laziness(self._dadg_key, lazily_evaluated=False)
        self._ctx.dadg.observe(self._dadg_key, "image_manager", self._observer_callback)

    def __del__(self):
        self._ctx.dadg.set_evaluation_laziness(self._dadg_key, lazily_evaluated=True)

    def _observer_callback(self, new_value: torch.Tensor) -> None:
        self._layer().data = new_value.cpu().numpy()


def add_fixed_image_layer(*, ctx: AppContext, dadg_key: str, colormap: str = "yellow") -> napari.layers.Layer | None:
    if dadg_key in viewer().layers:
        logger.warning(f"Layer '{dadg_key}' is already shown.")
        return None
    value = ctx.dadg.get(dadg_key, soft=True)
    if isinstance(value, Error):
        raise RuntimeError(f"Error softly getting '{dadg_key}' from DADG: {value.description}.")
    initial_image = value if isinstance(value, torch.Tensor) else torch.zeros((500, 500))
    layer = viewer().add_image(initial_image.cpu().numpy(), colormap=colormap, blending="additive",
                               interpolation2d="linear", name=dadg_key)
    layer.my_plugin = _FixedImageLayerManager(ctx=ctx, layer=layer, dadg_key=dadg_key)
    return layer
