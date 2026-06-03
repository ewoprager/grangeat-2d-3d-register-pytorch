import logging
import weakref
from typing import Callable

import napari.layers
import pandas as pd
import torch
import traitlets
from napari.utils.events import Event

from reg23_app.gui.viewer_singleton import viewer
from reg23_experiments.data.segmentation import OrderedPoints2D
from reg23_experiments.data.structs import Error
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph

__all__ = ["add_electrode_layer"]

logger = logging.getLogger(__name__)


class _ElectrodeLayerManager:
    def __init__(self, *, dadg: DirectedAcyclicDataGraph, layer: napari.layers.Points, dadg_key: str,
                 xray_uid_dadg_key: str):
        self._dadg = dadg
        self._layer: Callable[[], napari.layers.Points | None] = weakref.ref(layer)
        self._dadg_key = dadg_key
        self._xray_uid_dadg_key = xray_uid_dadg_key
        layer.events.connect(self._on_layer_change)

    def _update_dadg_from_layer(self) -> None:
        if (layer := self._layer()) is None:
            return
        try:
            electrode_points = OrderedPoints2D(data=torch.tensor(layer.data, dtype=torch.float64))
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


def add_electrode_layer(*, dadg: DirectedAcyclicDataGraph, namespace: str | None = None) -> napari.layers.Layer | None:
    dadg_key = "electrode_points" if namespace is None else f"{namespace}__electrode_points"
    if dadg_key in viewer().layers:
        logger.warning(f"Layer '{dadg_key}' is already shown.")
        return None
    uid_dadg_key = "xray_sop_instance_uid" if namespace is None else f"{namespace}__xray_sop_instance_uid"
    if isinstance(uid := dadg.get(uid_dadg_key), Error):
        logger.error(f"Failed to get X-ray UID for electrode layer.")
        return None
    if not isinstance(uid, str):
        logger.error(f"Expected UID to be a str, got: '{uid}'.")
        return None
    electrode_points: OrderedPoints2D | Error = dadg.get(dadg_key)
    if isinstance(electrode_points, Error):
        logger.error(f"Failed to get electrode point data for layer: {electrode_points.description}.")
        return None
    layer = viewer().add_points(  #
        electrode_points.data.numpy(),  #
        size=4.0,  #
        name=dadg_key,  #
        features=pd.DataFrame([{"label": f"{i}"} for i in range(1, electrode_points.count + 1)]),  #
        text={"string": "{label}", "size": 16}  #
    )
    layer.my_plugin = _ElectrodeLayerManager(dadg=dadg, layer=layer, dadg_key=dadg_key, xray_uid_dadg_key=uid_dadg_key)
    return layer
