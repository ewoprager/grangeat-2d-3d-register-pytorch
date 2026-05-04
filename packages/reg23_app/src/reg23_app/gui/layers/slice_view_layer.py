import logging
import weakref
from typing import Any

import napari.layers
import torch

from reg23_app.context import AppContext
from reg23_app.gui.viewer_singleton import viewer
from reg23_experiments.data.structs import Error
from reg23_experiments.ops.data_manager import dadg_updater

__all__ = ["add_slice_view_layer"]

logger = logging.getLogger(__name__)


@dadg_updater(names_returned=["ct_slice"])
def extract_ct_slice(*, untruncated_ct_volume: torch.Tensor, ct_slice_index: int) -> dict[str, Any]:
    clamped_index = max(0, min(untruncated_ct_volume.size()[0] - 1, ct_slice_index))
    return {"ct_slice": untruncated_ct_volume[clamped_index]}


class _SliceViewLayerManager:
    def __init__(self, *, ctx: AppContext, layer: napari.layers.Layer):
        self._ctx = ctx
        self._layer = weakref.ref(layer)
        #
        self._ctx.dadg.set("ct_slice_index", 0, check_equality=True)
        self._ctx.dadg.add_updater("extract_ct_slice", extract_ct_slice)
        self._ctx.dadg.set_evaluation_laziness("ct_slice", lazily_evaluated=False)
        self._layer().mouse_drag_callbacks.append(self._mouse_drag)
        self._ctx.dadg.observe("ct_slice", "ct_slice_manager", self._slice_observer_callback)

        self._index_observer_callback_loop_prevention = False
        self._ctx.dadg.observe("ct_slice_index", "ct_index_manager", self._index_observer_callback)

    def __del__(self):
        self._ctx.dadg.set_evaluation_laziness("ct_slice", lazily_evaluated=True)
        self._ctx.dadg.remove_updater("extract_ct_slice")

    def _slice_observer_callback(self, new_value: torch.Tensor) -> None:
        self._layer().data = new_value.cpu().numpy()

    def _index_observer_callback(self, new_value: int) -> None:
        if self._index_observer_callback_loop_prevention:
            return
        self._index_observer_callback_loop_prevention = True
        max_index = self._ctx.dadg.get("untruncated_ct_volume").size()[0] - 1
        self._ctx.dadg.set("ct_slice_index", max(0, min(max_index, new_value)), check_equality=True)
        self._index_observer_callback_loop_prevention = False

    def _mouse_drag(self, layer, event):
        if event.button == 1 and self._ctx.input_manager.ctrl_pressed:  # Ctrl-left click drag
            # mouse down
            dragged = False
            drag_start = torch.tensor(event.position)
            index_start: int = self._ctx.dadg.get("ct_slice_index")
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True

                delta = self._ctx.state.gui_settings.slice_index_sensitivity * (
                        torch.tensor(event.position) - drag_start).flip((0,))
                self._ctx.dadg.set("ct_slice_index", index_start + int(round(delta[1].item())), check_equality=True)
                yield
            # on release
            if dragged:
                # dragged
                pass
            else:
                # just clicked
                pass


def add_slice_view_layer(*, ctx: AppContext) -> napari.layers.Layer | None:
    name = "ct_slice_view"
    if name in viewer().layers:
        logger.warning(f"Layer {name} is already shown.")
        return None
    value = ctx.dadg.get("untruncated_ct_volume", soft=True)
    if isinstance(value, Error):
        raise RuntimeError(f"Error softly getting 'untruncated_ct_volume' from DADG: {value.description}.")
    initial_image = value[0] if isinstance(value, torch.Tensor) else torch.zeros((500, 500))
    layer = viewer().add_image(initial_image.cpu().numpy(), colormap="blue", blending="additive",
                               interpolation2d="linear", name=name)
    layer.my_plugin = _SliceViewLayerManager(ctx=ctx, layer=layer)
    return layer
