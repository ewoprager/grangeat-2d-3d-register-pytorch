import logging
import weakref
from collections.abc import Callable

import napari.layers
import pandas as pd

from reg23_app.context import AppContext
from reg23_app.gui.viewer_singleton import viewer
from reg23_experiments.data.segmentation import OrderedPoints3D
from reg23_experiments.data.structs import Error

__all__ = ["add_transformed_electrodes_layer"]

logger = logging.getLogger(__name__)


class _TransformedElectrodesLayerManager:
    def __init__(self, *, ctx: AppContext, layer: napari.layers.Points):
        logger.debug(f"Initializing _TransformedElectrodesLayerManager")
        self._ctx = ctx
        self._layer: Callable[[], napari.layers.Points | None] = weakref.ref(layer)
        #
        self._electrodes_key = "transformed_micro_electrodes"
        self._ctx.dadg.set_evaluation_laziness(self._electrodes_key, lazily_evaluated=False)
        self._ctx.dadg.observe(self._electrodes_key, "layer", self._observer_callback)

    def __del__(self):
        self._ctx.dadg.set_evaluation_laziness(self._electrodes_key, lazily_evaluated=True)

    def _observer_callback(self, new_value: OrderedPoints3D) -> None:
        if (layer := self._layer()) is not None:
            layer.features = pd.DataFrame([{"label": f"{i}"} for i in range(1, new_value.count + 1)])
            layer.data = new_value.data.flip(dims=(1,)).cpu().numpy()
        else:
            logger.warning(f"No layer to display transformed_electrodes.")


def add_transformed_electrodes_layer(*, ctx: AppContext) -> napari.layers.Points | None:
    logger.debug(f"Adding transformed electrodes layer")
    electrodes_key = "transformed_micro_electrodes"
    if electrodes_key in viewer().layers:
        logger.warning(f"Layer '{electrodes_key}' is already shown.")
        return None
    points: OrderedPoints3D | Error = ctx.dadg.get(electrodes_key, soft=True)
    if isinstance(points, Error):
        raise RuntimeError(f"Error softly getting '{electrodes_key}' from DADG: {points.description}.")
    initial_points = points.data.flip(dims=(1,))
    logger.debug(f"Adding transformed electrodes layer '{electrodes_key}' to napari viewer")
    layer = viewer().add_points(  #
        initial_points.cpu().numpy(),  #
        ndim=3,  #
        size=1.0,  #
        symbol="diamond",  #
        name=electrodes_key,  #
        features=pd.DataFrame([{"label": f"{i}"} for i in range(1, points.count + 1)]),  #
        text={"string": "{label}", "size": 16}  #
    )
    layer.my_plugin = _TransformedElectrodesLayerManager(ctx=ctx, layer=layer)
    return layer
