import logging
import weakref
from typing import Callable

import napari.layers
import pandas as pd
import torch
from napari.layers.base import ActionType
from napari.utils.events import Event

from reg23_app.context import AppContext
from reg23_app.gui.viewer_singleton import viewer
from reg23_experiments.data.structs import Error

__all__ = ["add_electrode_layer"]

logger = logging.getLogger(__name__)


class _ElectrodeLayerManager:
    def __init__(self, *, ctx: AppContext, layer: napari.layers.Points, dadg_key: str, xray_uid_dadg_key: str):
        self._ctx = ctx
        self._layer: Callable[[], napari.layers.Points | None] = weakref.ref(layer)
        self._dadg_key = dadg_key
        self._xray_uid_dadg_key = xray_uid_dadg_key
        if (layer := self._layer()) is not None:
            layer.events.connect(self._on_layer_change)
        else:
            logger.error("Failed to find layer for initialisation of ElectrodeLayerManager.")

    def _on_layer_change(self, event: Event):
        if event.type != "data":
            return
        if (layer := self._layer()) is None:
            return
        if isinstance(uid := self._ctx.dadg.get(self._xray_uid_dadg_key), Error):
            logger.error(f"Failed to get X-ray UID on electrode layer change: {uid.description}")
            return
        if not isinstance(uid, str):
            logger.error(f"Expected UID to be a str, got: '{uid}'.")
            return
        if event.action == ActionType.ADDED:
            tensor = torch.tensor(layer.data)
            layer.features = pd.DataFrame([{"label": f"{i}"} for i in range(1, tensor.size()[0] + 1)])
            self._ctx.dadg.set(self._dadg_key, tensor)
            res = self._ctx.electrode_save_manager.add(uid, tensor[-1])
            if isinstance(res, Error):
                logger.error(f"Error saving electrode point data: {res.description}")
        elif event.action == ActionType.CHANGED:
            tensor = torch.tensor(layer.data)
            self._ctx.dadg.set(self._dadg_key, tensor)
            for index in event.data_indices:
                res = self._ctx.electrode_save_manager.move(  #
                    uid=uid,  #
                    index=int(index),  #
                    tensor=tensor[int(index)]  #
                )
                if isinstance(res, Error):
                    logger.error(f"Error saving electrode point data: {res.description}")
        elif event.action == ActionType.REMOVING:
            for index in event.data_indices:
                res = self._ctx.electrode_save_manager.remove(  #
                    uid=uid,  #
                    index=int(index)  #
                )
                if isinstance(res, Error):
                    logger.error(f"Error saving electrode point data: {res.description}")
        elif event.action == ActionType.REMOVED:
            tensor = torch.tensor(layer.data)
            layer.features = pd.DataFrame([{"label": f"{i}"} for i in range(1, tensor.size()[0] + 1)])
            self._ctx.dadg.set(self._dadg_key, tensor)


def add_electrode_layer(*, ctx: AppContext, namespace: str | None = None) -> napari.layers.Layer | None:
    dadg_key = "electrode_points" if namespace is None else f"{namespace}__electrode_points"
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
    tensor = ctx.electrode_save_manager.get(uid)  # ToDo: Move the management of the data to a manager class
    if tensor is None:
        layer = viewer().add_points(  #
            ndim=2,  #
            size=4.0,  #
            name=dadg_key,  #
            features=pd.DataFrame(columns=["label"]),  #
            text={"string": "{label}", "size": 16}  #
        )
    else:
        layer = viewer().add_points(  #
            tensor.numpy(),  #
            size=4.0,  #
            name=dadg_key,  #
            features=pd.DataFrame([{"label": f"{i}"} for i in range(1, tensor.size()[0] + 1)]),  #
            text={"string": "{label}", "size": 16}  #
        )
    ctx.dadg.set(dadg_key, tensor)
    layer.my_plugin = _ElectrodeLayerManager(ctx=ctx, layer=layer, dadg_key=dadg_key, xray_uid_dadg_key=uid_dadg_key)
    return layer
