import logging
import weakref
from typing import Callable

import pandas as pd
from magicgui.widgets import request_values
import napari.layers
from napari.layers.base import ActionType
from napari.utils.events import Event
import torch
import numpy as np
import pprint

from reg23_app.context import AppContext
from reg23_app.gui.viewer_singleton import viewer

from reg23_experiments.data.structs import Error

__all__ = ["add_ct_fiducial_layer"]

logger = logging.getLogger(__name__)


class _CTFiducialLayerManager:
    def __init__(self, *, ctx: AppContext, layer: napari.layers.Points):
        self._ctx = ctx
        self._layer: Callable[[], napari.layers.Points | None] = weakref.ref(layer)
        if (layer := self._layer()) is not None:
            layer.events.connect(self._on_layer_change)
        else:
            logger.error("Failed to find layer for initialisation of CTFiducialLayerManager.")

    def _on_layer_change(self, event: Event):
        if event.type != "data":
            return
        if (layer := self._layer()) is None:
            return
        if isinstance(uid := self._ctx.dadg.get("ct_series_uid"), Error):
            logger.error(f"Failed to get CT UID on fiducial layer change: {uid.description}")
            return
        if event.action == ActionType.ADDED:
            # Get a name for the new point
            name: str | None = None
            while not name:
                values = request_values(name={"annotation": str, "label": "Enter a unique name for the point"})
                if values["name"]:
                    name = values["name"]
            new_features = layer.features.copy()
            new_features.iloc[-1]["label"] = name
            layer.features = new_features
            # Save the data
            res = self._ctx.ct_fiducial_save_manager.set(  #
                uid=uid,  #
                name=name,  #
                value=torch.tensor(event.value[-1])  #
            )
            if isinstance(res, Error):
                logger.error(f"Error saving fiducial point data: {res.description}")
        elif event.action == ActionType.CHANGED:
            for index in event.data_indices:
                res = self._ctx.ct_fiducial_save_manager.set(  #
                    uid=uid,  #
                    name=layer.features.at[int(index), "label"],  #
                    value=torch.tensor(event.value[int(index)])  #
                )
                if isinstance(res, Error):
                    logger.error(f"Error saving fiducial point data: {res.description}")
        elif event.action == ActionType.REMOVING:
            for index in event.data_indices:
                res = self._ctx.ct_fiducial_save_manager.remove(  #
                    uid=uid,  #
                    name=layer.features.at[int(index), "label"]  #
                )
                if isinstance(res, Error):
                    logger.error(f"Error saving fiducial point data: {res.description}")


def add_ct_fiducial_layer(*, ctx: AppContext) -> napari.layers.Layer | None:
    if "ct_fiducial_points" in viewer().layers:
        logger.warning(f"Layer 'ct_fiducial_points' is already shown.")
        return None
    res = ctx.ct_fiducial_save_manager.get(ctx.dadg.get("ct_series_uid"))
    if res is None:
        layer = viewer().add_points(  #
            ndim=3,  #
            size=8.0,  #
            name="ct_fiducial_points",  #
            features=pd.DataFrame(columns=["label"]),  #
            text={"string": "{label}", "size": 16, "color": "white"}  #
        )
    else:
        names, tensor = res
        layer = viewer().add_points(  #
            tensor.numpy(),  #
            size=8.0,  #
            name="ct_fiducial_points",  #
            features=pd.DataFrame([{"label": name} for name in names]),  #
            text={"string": "{label}", "size": 16, "color": "white"}  #
        )
    # ctx.dadg.set(dadg_key, tensor)
    layer.my_plugin = _CTFiducialLayerManager(ctx=ctx, layer=layer)
    return layer
