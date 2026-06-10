import logging
import weakref
from typing import Callable

import napari.layers
import pandas as pd
import torch
import traitlets
from napari.layers.base import ActionType
from napari.utils.events import Event

from reg23_app.gui.floating import get_string_required
from reg23_app.gui.viewer_singleton import viewer
from reg23_experiments.data.segmentation import NamedPoints2D
from reg23_experiments.data.structs import Error
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph

__all__ = ["add_xray_fiducial_layer"]

logger = logging.getLogger(__name__)


class _XRayFiducialLayerManager:
    def __init__(self, *, dadg: DirectedAcyclicDataGraph, layer: napari.layers.Points, dadg_key: str,
                 xray_uid_dadg_key: str):
        self._dadg = dadg
        self._layer: Callable[[], napari.layers.Points | None] = weakref.ref(layer)
        self._dadg_key = dadg_key
        self._xray_uid_dadg_key = xray_uid_dadg_key
        layer.events.connect(self._on_layer_change)
        self._dadg.observe(self._dadg_key, "layer", self._update_layer_from_dadg)
        self._callback_loop_prevention: bool = False

    def _update_layer_from_dadg(self, new_value: NamedPoints2D) -> None:
        if self._callback_loop_prevention:
            return
        if (layer := self._layer()) is None:
            return
        self._callback_loop_prevention = True
        layer.features = pd.DataFrame([{"label": name} for name in new_value.names])
        layer.data = new_value.data.flip(dims=(1,)).numpy()
        self._callback_loop_prevention = False

    def _update_dadg_from_layer(self) -> None:
        if self._callback_loop_prevention:
            return
        if (layer := self._layer()) is None:
            return
        names = layer.features["label"].values.tolist()
        data = torch.tensor(layer.data).flip(dims=(1,))
        try:
            fiducial_points = NamedPoints2D(names=names, data=data)
        except traitlets.TraitError as e:
            logger.error(f"Error converting fiducial point data from layer: {e}")
            return
        self._callback_loop_prevention = True
        err = self._dadg.set(self._dadg_key, fiducial_points)
        if isinstance(err, Error):
            logger.error(f"Error processing X-ray fiducial point data: {err.description}")
        self._callback_loop_prevention = False

    def _on_layer_change(self, event: Event):
        if self._callback_loop_prevention:
            return
        if event.type != "data":
            return
        if (layer := self._layer()) is None:
            return
        if event.action == ActionType.ADDED:
            # Get a name for the new point
            def validator(_name: str) -> None | Error:
                if _name in layer.features["label"].values.tolist():
                    return Error(f"Name '{_name}' already in use.")
                return None

            name = get_string_required(message="Enter a unique name for the point", validator=validator)
            new_features = layer.features.copy()
            new_features.iloc[-1]["label"] = name
            layer.features = new_features
        elif event.action == ActionType.REMOVED:
            layer.features = layer.features.head(layer.data.shape[0])

        self._update_dadg_from_layer()


def add_xray_fiducial_layer(*, dadg: DirectedAcyclicDataGraph,
                            namespace: str | None = None) -> napari.layers.Layer | None:
    dadg_key = "fiducial_points" if namespace is None else f"{namespace}__fiducial_points"
    if dadg_key in viewer().layers:
        logger.warning(f"Layer '{dadg_key}' is already shown.")
        return None
    uid_dadg_key = "xray_sop_instance_uid" if namespace is None else f"{namespace}__xray_sop_instance_uid"
    uid: str | Error = dadg.get(uid_dadg_key)
    if isinstance(uid, Error):
        logger.error(f"Failed to get X-ray UID for electrode layer.")
        return None
    res: NamedPoints2D | Error = dadg.get(dadg_key)
    if isinstance(res, Error):
        logger.error(f"Error getting fiducial point data for layer: {res.description}")
        return None
    if res.names:
        features = pd.DataFrame([{"label": name} for name in res.names])
    else:
        features = pd.DataFrame(columns=["label"])
    layer = viewer().add_points(  #
        res.data.flip(dims=(1,)).numpy(),  #
        size=8.0,  #
        name=dadg_key,  #
        features=features,  #
        text={"string": "{label}", "size": 16}  #
    )
    layer.my_plugin = _XRayFiducialLayerManager(dadg=dadg, layer=layer, dadg_key=dadg_key,
                                                xray_uid_dadg_key=uid_dadg_key)
    return layer
