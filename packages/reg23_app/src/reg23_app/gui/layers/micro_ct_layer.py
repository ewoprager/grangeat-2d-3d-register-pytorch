import logging
import weakref
from typing import Callable

import napari.layers
import zarr

from reg23_app.context import AppContext
from reg23_app.gui.viewer_singleton import viewer
from reg23_experiments.data.structs import Error

__all__ = ["add_micro_ct_layer"]

logger = logging.getLogger(__name__)


class _MicroCTLayerManager:
    def __init__(self, *, ctx: AppContext, layer: napari.layers.Image):
        self._ctx = ctx
        self._layer: Callable[[], napari.layers.Image | None] = weakref.ref(layer)

        self._ctx.state.observe(self._micro_ct_changed, names=["micro_ct_slice_path"])

    def _micro_ct_changed(self, *args) -> None:
        if (layer := self._layer()) is None:
            return
        res: zarr.Array | None | Error = self._ctx.zarr_manager.get_micro_ct()
        if isinstance(res, Error):
            logger.error(f"Failed to load microCT for layer: {res.description}")
            return
        if res is None:
            logger.warning(f"Cannot open microCT layer; no microCT selected.")
            return
        layer.data = res


def add_micro_ct_layer(*, ctx: AppContext) -> napari.layers.Image | None:
    name = "micro_ct_view"
    if name in viewer().layers:
        logger.warning(f"Layer {name} is already shown.")
        return None

    res: zarr.Array | None | Error = ctx.zarr_manager.get_micro_ct()
    if isinstance(res, Error):
        logger.error(f"Failed to load microCT for layer: {res.description}")
        return None
    if res is None:
        logger.warning(f"Cannot open microCT layer; no microCT selected.")
        return None

    layer: napari.layers.Image = viewer().add_image(  #
        res,  #
        colormap="blue",  #
        blending="additive",  #
        interpolation2d="linear",  #
        name=name,  #
    )
    layer.my_plugin = _MicroCTLayerManager(ctx=ctx, layer=layer)
    return layer
