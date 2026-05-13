import logging
import weakref
from typing import Callable

import napari.layers
import pandas as pd
import torch
from magicgui.widgets import request_values
from napari.layers.base import ActionType
from napari.utils.events import Event

from reg23_app.context import AppContext
from reg23_app.gui.viewer_singleton import viewer
from reg23_experiments.data.structs import Error

__all__ = ["add_ct_fiducial_layer"]

logger = logging.getLogger(__name__)


class _CTFiducialLayerManager:
    def __init__(self, *, ctx: AppContext, layer: napari.layers.Points):
        self._ctx = ctx
        self._layer: Callable[[], napari.layers.Points | None] = weakref.ref(layer)
        layer.events.connect(self._on_layer_change)

    def _on_layer_change(self, event: Event):
        if event.type != "data":
            return
        if (layer := self._layer()) is None:
            return
        if isinstance(uid := self._ctx.dadg.get("ct_series_uid"), Error):
            logger.error(f"Failed to get CT UID on fiducial layer change: {uid.description}")
            return
        if not isinstance(uid, str):
            logger.error(f"Expected UID to be a str, got: '{uid}'.")
            return
        if event.action == ActionType.ADDED:
            # Get a name for the new point
            message_prefix: str | None = None
            message_suffix: str = "Enter a unique name for the point"
            while True:
                prompt = message_suffix
                if message_prefix is not None:
                    prompt = message_prefix + ";\n" + prompt
                values = request_values(name={"annotation": str, "label": prompt})
                if not values:
                    message_prefix = "No values provided."
                    continue
                name = values["name"]
                if not name:
                    message_prefix = "No name provided."
                    continue
                if name in layer.features["label"].values.tolist():
                    message_prefix = f"Name '{name}' already in use."
                    continue
                break

            new_features = layer.features.copy()
            new_features.iloc[-1]["label"] = name
            layer.features = new_features
            # Save the data
            res = self._ctx.ct_fiducial_save_manager.set(  #
                uid=uid,  #
                name=name,  #
                value=torch.tensor(event.value[-1]).flip(dims=(0,))  #
            )
            if isinstance(res, Error):
                logger.error(f"Error saving fiducial point data: {res.description}")
            self._ctx.dadg.set("ct_fiducial_points",
                               (layer.features["label"].values.tolist(), torch.tensor(layer.data).flip(dims=(1,))))
        elif event.action == ActionType.CHANGED:
            for index in event.data_indices:
                res = self._ctx.ct_fiducial_save_manager.set(  #
                    uid=uid,  #
                    name=layer.features.at[int(index), "label"],  #
                    value=torch.tensor(event.value[int(index)]).flip(dims=(0,))  #
                )
                if isinstance(res, Error):
                    logger.error(f"Error saving fiducial point data: {res.description}")
            self._ctx.dadg.set("ct_fiducial_points",
                               (layer.features["label"].values.tolist(), torch.tensor(layer.data).flip(dims=(1,))))
        elif event.action == ActionType.REMOVING:
            for index in event.data_indices:
                res = self._ctx.ct_fiducial_save_manager.remove(  #
                    uid=uid,  #
                    name=layer.features.at[int(index), "label"]  #
                )
                if isinstance(res, Error):
                    logger.error(f"Error saving fiducial point data: {res.description}")
        elif event.action == ActionType.REMOVED:
            layer.features = layer.features.head(layer.data.shape[0])
            self._ctx.dadg.set("ct_fiducial_points",
                               (layer.features["label"].values.tolist(), torch.tensor(layer.data).flip(dims=(1,))))


def add_ct_fiducial_layer(*, ctx: AppContext) -> napari.layers.Layer | None:
    if "ct_fiducial_points" in viewer().layers:
        logger.warning(f"Layer 'ct_fiducial_points' is already shown.")
        return None
    if isinstance(uid := ctx.dadg.get("ct_series_uid"), Error):
        logger.error(f"Failed to get CT UID for fiducial layer.")
        return None
    if not isinstance(uid, str):
        logger.error(f"Expected UID to be a str, got: '{uid}'.")
        return None
    res: tuple[list[str], torch.Tensor] | None | Error = ctx.dadg.get("ct_fiducial_points")
    if isinstance(res, Error):
        logger.error(f"Failed to get CT fiducial point data for layer: {res.description}")
        return None
    if res is None:
        layer = viewer().add_points(  #
            ndim=3,  #
            size=8.0,  #
            name="ct_fiducial_points",  #
            features=pd.DataFrame(columns=["label"]),  #
            text={"string": "{label}", "size": 16}  #
        )
    else:
        names, tensor = res
        layer = viewer().add_points(  #
            tensor.flip(dims=(1,)).numpy(),  #
            size=8.0,  #
            name="ct_fiducial_points",  #
            features=pd.DataFrame([{"label": name} for name in names]),  #
            text={"string": "{label}", "size": 16}  #
        )
    layer.my_plugin = _CTFiducialLayerManager(ctx=ctx, layer=layer)
    return layer
