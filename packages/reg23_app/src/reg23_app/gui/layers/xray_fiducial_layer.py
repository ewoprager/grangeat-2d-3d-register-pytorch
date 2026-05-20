import logging
import weakref
from typing import Callable

import napari.layers
import pandas as pd
import torch
from napari.layers.base import ActionType
from napari.utils.events import Event

from reg23_app.context import AppContext
from reg23_app.gui.floating import get_string_required
from reg23_app.gui.viewer_singleton import viewer
from reg23_experiments.data.structs import Error

__all__ = ["add_xray_fiducial_layer"]

logger = logging.getLogger(__name__)


class _XRayFiducialLayerManager:
    def __init__(self, *, ctx: AppContext, layer: napari.layers.Points, dadg_key: str, xray_uid_dadg_key: str):
        self._ctx = ctx
        self._layer: Callable[[], napari.layers.Points | None] = weakref.ref(layer)
        self._dadg_key = dadg_key
        self._xray_uid_dadg_key = xray_uid_dadg_key
        layer.events.connect(self._on_layer_change)
        self._ctx.dadg.observe(self._dadg_key, "layer", self._update_layer_from_dadg)
        self._callback_loop_prevention: bool = False

    def _update_layer_from_dadg(self, new_value: tuple[list[str], torch.Tensor]) -> None:
        if self._callback_loop_prevention:
            return
        if (layer := self._layer()) is None:
            return
        new_names, new_points = new_value
        self._callback_loop_prevention = True
        layer.features = pd.DataFrame([{"label": name} for name in new_names])
        layer.data = new_points.flip(dims=(1,)).numpy()
        self._callback_loop_prevention = False

    def _update_dadg_from_layer(self) -> None:
        if self._callback_loop_prevention:
            return
        if (layer := self._layer()) is None:
            return
        names = layer.features["label"].values.tolist()
        points = torch.tensor(layer.data).flip(dims=(1,))
        self._callback_loop_prevention = True
        self._ctx.dadg.set(self._dadg_key, (names, points))
        self._callback_loop_prevention = False

    def _on_layer_change(self, event: Event):
        if self._callback_loop_prevention:
            return
        if event.type != "data":
            return
        if (layer := self._layer()) is None:
            return
        if isinstance(uid := self._ctx.dadg.get(self._xray_uid_dadg_key), Error):
            logger.error(f"Failed to get X-ray UID on fiducial layer change: {uid.description}")
            return
        if not isinstance(uid, str):
            logger.error(f"Expected UID to be a str, got: '{uid}'.")
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
        # Save the data
        # ToDo: Move this to the ParamDADGParityManager or similar so that fiducial refinement gets saved
        res = self._ctx.xray_fiducial_save_manager.set(  #
            uid=uid,  #
            names=layer.features["label"].tolist(),  #
            points=torch.tensor(layer.data).flip(dims=(1,))  #
        )
        if isinstance(res, Error):
            logger.error(f"Error saving fiducial point data: {res.description}")
        self._update_dadg_from_layer()


def add_xray_fiducial_layer(*, ctx: AppContext, namespace: str | None = None) -> napari.layers.Layer | None:
    dadg_key = "fiducial_points" if namespace is None else f"{namespace}__fiducial_points"
    if dadg_key in viewer().layers:
        logger.warning(f"Layer '{dadg_key}' is already shown.")
        return None
    uid_dadg_key = "xray_sop_instance_uid" if namespace is None else f"{namespace}__xray_sop_instance_uid"
    if isinstance(uid := ctx.dadg.get(uid_dadg_key), Error):
        logger.error(f"Failed to get X-ray UID for electrode layer.")
        return None
    if not isinstance(uid, str):
        logger.error(f"Expected UID to be a str, got: '{uid}'.")
        return None
    res: tuple[list[str], torch.Tensor] | None | Error = ctx.dadg.get(dadg_key)
    if isinstance(res, Error):
        logger.error(f"Error getting fiducial point data for layer: {res.description}")
        return None
    if res is None:
        layer = viewer().add_points(  #
            ndim=2,  #
            size=8.0,  #
            name=dadg_key,  #
            features=pd.DataFrame(columns=["label"]),  #
            text={"string": "{label}", "size": 16}  #
        )
    else:
        names, tensor = res
        layer = viewer().add_points(  #
            tensor.flip(dims=(1,)).numpy(),  #
            size=8.0,  #
            name=dadg_key,  #
            features=pd.DataFrame([{"label": name} for name in names]),  #
            text={"string": "{label}", "size": 16}  #
        )
    layer.my_plugin = _XRayFiducialLayerManager(ctx=ctx, layer=layer, dadg_key=dadg_key, xray_uid_dadg_key=uid_dadg_key)
    return layer
