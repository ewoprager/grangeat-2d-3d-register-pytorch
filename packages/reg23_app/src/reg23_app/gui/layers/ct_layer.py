import logging
import weakref

import napari.layers
import torch

from reg23_app.context import AppContext
from reg23_app.gui.viewer_singleton import viewer
from reg23_experiments.data.structs import Error

__all__ = ["add_ct_layer"]

logger = logging.getLogger(__name__)


class _CTLayerManager:
    def __init__(self, *, ctx: AppContext, layer: napari.layers.Image):
        self._ctx = ctx
        self._layer = weakref.ref(layer)

        self._ctx.dadg.observe("untruncated_ct_volume", "ct_layer", self._observer_callback)
        self._ctx.dadg.set_evaluation_laziness("untruncated_ct_volume", lazily_evaluated=False)
        self._ctx.dadg.observe("ct_spacing", "ct_layer", self._spacing_callback)
        self._ctx.dadg.set_evaluation_laziness("ct_spacing", lazily_evaluated=False)

    def __del__(self):
        self._ctx.dadg.set_evaluation_laziness("untruncated_ct_volume", lazily_evaluated=True)
        self._ctx.dadg.set_evaluation_laziness("ct_spacing", lazily_evaluated=True)

    def _observer_callback(self, new_value: torch.Tensor) -> None:
        self._layer().data = new_value.cpu().numpy()

    def _spacing_callback(self, new_value: torch.Tensor) -> None:
        self._layer().scale = new_value.flip(dims=(0,)).cpu().numpy()


def add_ct_layer(*, ctx: AppContext) -> napari.layers.Image | None:
    name = "ct_view"
    if name in viewer().layers:
        logger.warning(f"Layer {name} is already shown.")
        return None
    value = ctx.dadg.get("untruncated_ct_volume", soft=True)
    spacing = ctx.dadg.get("ct_spacing", soft=True)
    if isinstance(value, Error):
        raise RuntimeError(f"Error softly getting 'untruncated_ct_volume' from DADG: {value.description}.")
    if isinstance(spacing, Error):
        raise RuntimeError(f"Error softly getting 'ct_spacing' from DADG: {spacing.description}.")
    initial_image = value if isinstance(value, torch.Tensor) else torch.zeros((200, 500, 500))
    layer: napari.layers.Image = viewer().add_image(initial_image.cpu().numpy(), colormap="blue", blending="additive",
                                                    interpolation2d="linear", name=name, scale=spacing.flip(dims=(0,)).cpu().numpy())
    layer.my_plugin = _CTLayerManager(ctx=ctx, layer=layer)
    return layer
