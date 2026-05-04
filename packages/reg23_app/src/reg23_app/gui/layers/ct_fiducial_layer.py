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
        if (layer := self._layer()) is None:
            return
        if event.type == "data":
            if False:
                logger.info(f"action value: {event.action.value}")
                logger.info(f"data indices: {event.data_indices}")
                logger.info(f"index: {event.index}")
                logger.info(f"vertex indices: {event.vertex_indices}")
                return
            if event.action == ActionType.ADDED:
                name = None
                while not name:
                    values = request_values(name={"annotation": str, "label": "Enter a unique name for the point"})
                    if values["name"]:
                        name = values["name"]
                new_features = layer.features.copy()
                new_features.iloc[-1]["label"] = name
                layer.features = new_features
            elif event.action == ActionType.CHANGED:
                for index in event.data_indices:
                    self._ctx.ct_fiducial_save_manager.move(uid=self._ctx.dadg.get("ct_series_uid"),
                                                            name=layer.features.at[index, "label"],
                                                            value=torch.tensor(event.value[index]))

            #

            # tensor = torch.tensor(self._layer().data)

            # self._layer().text.values = [f"{i}" for i in range(1, tensor.size()[0] + 1)]

            # self._ctx.dadg.set(self._dadg_key, tensor)

            # res = self._ctx.ct_fiducials_save_manager.set(self._ctx.dadg.get(self._xray_uid_dadg_key), tensor)

            # if isinstance(res, Error):

            #     logger.error(f"Error saving electrode point data: {res.description}")


def add_ct_fiducial_layer(*, ctx: AppContext) -> napari.layers.Layer | None:
    if "ct_fiducial_points" in viewer().layers:
        logger.warning(f"Layer 'ct_fiducial_points' is already shown.")
        return None
    res = ctx.ct_fiducial_save_manager.get(ctx.dadg.get("ct_series_uid"))
    if res is None:
        layer = viewer().add_points(  #
            ndim=2,  #
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
