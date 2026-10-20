import logging
import weakref
from collections.abc import Callable

import napari.layers
import pandas as pd
import torch
import traitlets
from napari.utils.events import Event

from reg23_app.gui.viewer_singleton import viewer
from reg23_experiments.data.segmentation import OrderedPoints3D
from reg23_experiments.data.structs import Error
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph

__all__ = ["add_ct_electrode_layer"]

logger = logging.getLogger(__name__)


class _CTElectrodeLayerManager:
    def __init__(  #
        self,  #
        *,  #
        dadg: DirectedAcyclicDataGraph,  #
        layer: napari.layers.Points,  #
        dadg_key: str,  #
        ct_uid_dadg_key: str,  #
    ):
        self._dadg = dadg
        self._layer: Callable[[], napari.layers.Points | None] = weakref.ref(layer)
        self._dadg_key = dadg_key
        self._ct_uid_dadg_key = ct_uid_dadg_key
        layer.events.connect(self._on_layer_change)

    def _update_dadg_from_layer(self) -> None:
        if (layer := self._layer()) is None:
            return
        try:
            electrode_points = OrderedPoints3D(  #
                data=torch.tensor(layer.data, dtype=torch.float64).flip(dims=(1,)),  #
            )
        except traitlets.TraitError as e:
            logger.error(f"Error converting electrode point data from layer: {e}")
            return
        err = self._dadg.set(self._dadg_key, electrode_points)
        if isinstance(err, Error):
            logger.error(f"Error processing electrode point data: {err.description}")

    def _on_layer_change(self, event: Event):
        if event.type != "data":
            return
        if (layer := self._layer()) is None:
            return
        layer.features = pd.DataFrame([{"label": f"{i}"} for i in range(1, len(layer.data) + 1)])
        self._update_dadg_from_layer()


def add_ct_electrode_layer(  #
    *,  #
    dadg: DirectedAcyclicDataGraph,  #
    dadg_key: str,  #
    image_uid_dadg_key: str,  #
) -> napari.layers.Layer | None:
    if dadg_key in viewer().layers:
        logger.warning(f"Layer '{dadg_key}' is already shown.")
        return None
    if isinstance(uid := dadg.get(image_uid_dadg_key), Error):
        logger.error(f"Failed to get CT UID for electrode layer.")
        return None
    if not isinstance(uid, str):
        logger.error(f"Expected UID to be a str, got: '{uid}'.")
        return None
    electrode_points: OrderedPoints3D | Error = dadg.get(dadg_key)
    if isinstance(electrode_points, Error):
        logger.error(f"Failed to get electrode point data for layer: {electrode_points.description}.")
        return None
    layer = viewer().add_points(  #
        electrode_points.data.flip(dims=(1,)).numpy(),  #
        size=1.0,  #
        symbol="diamond",  #
        name=dadg_key,  #
        features=pd.DataFrame([{"label": f"{i}"} for i in range(1, electrode_points.count + 1)]),  #
        text={"string": "{label}", "size": 16},  #
    )
    layer.my_plugin = _CTElectrodeLayerManager(  #
        dadg=dadg,  #
        layer=layer,  #
        dadg_key=dadg_key,  #
        ct_uid_dadg_key=image_uid_dadg_key,  #
    )
    return layer
