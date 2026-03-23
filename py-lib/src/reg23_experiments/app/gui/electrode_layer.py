import logging
import weakref

import napari.layers
import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.app.context import AppContext
from reg23_experiments.app.gui.viewer_singleton import viewer

__all__ = ["add_electrode_layer"]

logger = logging.getLogger(__name__)


class _ElectrodeLayerManager:
    def __init__(self, *, ctx: AppContext, layer: napari.layers.Layer, dadg_key: str, xray_uid_dadg_key: str):
        self._ctx = ctx
        self._layer = weakref.ref(layer)
        self._dadg_key = dadg_key
        self._xray_uid_dadg_key = xray_uid_dadg_key
        self._layer().events.connect(self._on_layer_change)

    def _on_layer_change(self, event):
        if event.type == "data":
            tensor = torch.tensor(self._layer().data)
            self._layer().text.values = [f"{i}" for i in range(1, tensor.size()[0] + 1)]
            self._ctx.dadg.set(self._dadg_key, tensor)
            res = self._ctx.electrode_save_manager.set(self._ctx.dadg.get(self._xray_uid_dadg_key), tensor)
            if isinstance(res, Error):
                logger.error(f"Error saving electrode point data: {res.description}")


def add_electrode_layer(*, ctx: AppContext, namespace: str | None = None) -> napari.layers.Layer:
    uid_dadg_key = "xray_sop_instance_uid" if namespace is None else f"{namespace}__xray_sop_instance_uid"
    dadg_key = "electrode_points" if namespace is None else f"{namespace}__electrode_points"
    tensor = ctx.electrode_save_manager.get(ctx.dadg.get(uid_dadg_key))
    if tensor is None:
        layer = viewer().add_points(ndim=2, size=4.0, name=dadg_key)
    else:
        layer = viewer().add_points(tensor.numpy(), size=4.0,
                                    name="electrodes" if namespace is None else f"{namespace} electrodes")
        layer.text.values = [f"{i}" for i in range(1, tensor.size()[0] + 1)]
    ctx.dadg.set(dadg_key, tensor)
    layer.my_plugin = _ElectrodeLayerManager(ctx=ctx, layer=layer, dadg_key=dadg_key, xray_uid_dadg_key=uid_dadg_key)
    return layer
