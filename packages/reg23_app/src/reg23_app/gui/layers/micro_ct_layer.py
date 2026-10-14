import logging
import weakref
from typing import Callable

import napari.layers
import numpy as np
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

        self._ctx.dadg.observe("micro_ct_array", "layer", self._micro_ct_changed)
        self._ctx.dadg.set_evaluation_laziness("micro_ct_array", lazily_evaluated=False)

    def __del__(self):
        self._ctx.dadg.set_evaluation_laziness("micro_ct_array", lazily_evaluated=True)

    def _micro_ct_changed(self, new_value: zarr.Array | None) -> None:
        if (layer := self._layer()) is None:
            return
        if new_value is None:
            logger.warning("MicroCT no longer available to render in layer.")
            layer.data = None
            return
        layer.data = new_value
        layer.scale = np.flip(np.array(new_value.attrs["spacing"]))


def add_micro_ct_layer(*, ctx: AppContext) -> napari.layers.Image | None:
    name = "micro_ct_view"
    if name in viewer().layers:
        logger.warning(f"Layer {name} is already shown.")
        return None

    res: zarr.Array | None | Error = ctx.dadg.get("micro_ct_array")
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
        scale=res.attrs["spacing"],  #
    )
    layer.my_plugin = _MicroCTLayerManager(ctx=ctx, layer=layer)
    return layer
