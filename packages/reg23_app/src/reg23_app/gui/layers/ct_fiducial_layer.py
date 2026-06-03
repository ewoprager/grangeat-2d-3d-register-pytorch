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
from reg23_experiments.data.segmentation import NamedPoints3D
from reg23_experiments.data.structs import Error
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph

__all__ = ["add_ct_fiducial_layer"]

logger = logging.getLogger(__name__)


class _CTFiducialLayerManager:
    def __init__(self, *, dadg: DirectedAcyclicDataGraph, layer: napari.layers.Points):
        self._dadg = dadg
        self._layer: Callable[[], napari.layers.Points | None] = weakref.ref(layer)
        layer.events.connect(self._on_layer_change)
        self._dadg.observe("ct_fiducial_points", "layer", self._update_layer_from_dadg)
        self._callback_loop_prevention: bool = False

    def _update_layer_from_dadg(self, new_value: NamedPoints3D) -> None:
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
            fiducial_points = NamedPoints3D(names=names, data=data)
        except traitlets.TraitError as e:
            logger.error(f"Error converting fiducial point data from layer: {e}")
            return
        self._callback_loop_prevention = True
        err = self._dadg.set("ct_fiducial_points", fiducial_points)
        if isinstance(err, Error):
            logger.error(f"Error processing CT fiducial point data: {err.description}")
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


def add_ct_fiducial_layer(*, dadg: DirectedAcyclicDataGraph) -> napari.layers.Layer | None:
    if "ct_fiducial_points" in viewer().layers:
        logger.warning(f"Layer 'ct_fiducial_points' is already shown.")
        return None
    if isinstance(uid := dadg.get("ct_series_uid"), Error):
        logger.error(f"Failed to get CT UID for fiducial layer.")
        return None
    if not isinstance(uid, str):
        logger.error(f"Expected UID to be a str, got: '{uid}'.")
        return None
    fiducial_points: NamedPoints3D | Error = dadg.get("ct_fiducial_points")
    if isinstance(fiducial_points, Error):
        logger.error(f"Failed to get CT fiducial point data for layer: {fiducial_points.description}")
        return None
    # ToDo: Move initialisation to the layer manager?
    layer = viewer().add_points(  #
        fiducial_points.data.flip(dims=(1,)).numpy(),  #
        size=8.0,  #
        name="ct_fiducial_points",  #
        features=pd.DataFrame([{"label": name} for name in fiducial_points.names]),  #
        text={"string": "{label}", "size": 16}  #
    )
    layer.my_plugin = _CTFiducialLayerManager(dadg=dadg, layer=layer)
    return layer
