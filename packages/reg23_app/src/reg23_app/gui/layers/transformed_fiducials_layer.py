import logging
import weakref
from collections.abc import Callable

import napari.layers
import numpy as np
import pandas as pd
import scipy
import torch

from reg23_app.context import AppContext
from reg23_app.gui.viewer_singleton import viewer
from reg23_experiments.data.segmentation import NamedPoints3D
from reg23_experiments.data.structs import Error, Transformation

__all__ = ["add_transformed_fiducials_layer"]

logger = logging.getLogger(__name__)


class _TransformedFiducialsLayerManager:
    def __init__(self, *, ctx: AppContext, layer: napari.layers.Points):
        logger.debug(f"Initializing _TransformedFiducialsLayerManager")
        self._ctx = ctx
        self._layer: Callable[[], napari.layers.Points | None] = weakref.ref(layer)
        #
        self._fiducials_key = "transformed_micro_fiducials"
        self._current_transformation_key = "micro_ct_transformation"
        self._ctx.dadg.set_evaluation_laziness(self._fiducials_key, lazily_evaluated=False)
        self._ctx.dadg.observe(self._fiducials_key, "layer", self._observer_callback)
        layer.mouse_drag_callbacks.append(self._mouse_drag)

    def __del__(self):
        self._ctx.dadg.set_evaluation_laziness(self._fiducials_key, lazily_evaluated=True)

    def _observer_callback(self, new_value: NamedPoints3D) -> None:
        if (layer := self._layer()) is not None:
            layer.features = pd.DataFrame([{"label": name} for name in new_value.names])
            layer.data = new_value.data.flip(dims=(1,)).cpu().numpy()
        else:
            logger.warning(f"No layer to display transformed_fiducials.")

    def _mouse_drag(self, layer, event):
        if event.button == 1 and self._ctx.input_manager.ctrl_pressed:  # Ctrl-left click drag
            # mouse down
            dragged = False
            drag_start = np.array([event.position[-1], -event.position[-2]])
            rotation_start = scipy.spatial.transform.Rotation.from_rotvec(
                rotvec=self._ctx.dadg.get(self._current_transformation_key).rotation.cpu().numpy())
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True

                delta = self._ctx.state.gui_settings.rotation_sensitivity * (
                        np.array([event.position[-1], -event.position[-2]]) - drag_start)
                euler_angles = [delta[1], delta[0], 0.0]
                rot_euler = scipy.spatial.transform.Rotation.from_euler(seq="xyz", angles=euler_angles)
                rot_combined = rot_euler * rotation_start
                prev = self._ctx.dadg.get(self._current_transformation_key)
                self._ctx.dadg.set(  #
                    self._current_transformation_key,  #
                    Transformation(  #
                        rotation=torch.tensor(  #
                            rot_combined.as_rotvec(),  #
                            device=prev.rotation.device,  #
                            dtype=prev.rotation.dtype  #
                        ),  #
                        translation=prev.translation  #
                    )  #
                )
                yield
            # on release
            if dragged:
                # dragged
                pass
            else:
                # just clicked
                pass
        elif event.button == 2 and self._ctx.input_manager.ctrl_pressed:  # Ctrl-right click drag
            # mouse down
            dragged = False
            drag_start = torch.tensor(event.position[-2:])
            # rotation_start = scipy.spatial.transform.Rotation.from_rotvec(transformation.rotation.cpu().numpy())
            translation_start = self._ctx.dadg.get(self._current_transformation_key).translation[0:2].cpu()
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True

                delta = self._ctx.state.gui_settings.translation_sensitivity * (
                        torch.tensor(event.position[-2:]) - drag_start).flip((0,))
                prev = self._ctx.dadg.get(self._current_transformation_key)
                tr = prev.translation
                tr[0:2] = (translation_start + delta).to(device=tr.device)
                self._ctx.dadg.set(  #
                    self._current_transformation_key,  #
                    Transformation(  #
                        translation=tr,  #
                        rotation=prev.rotation  #
                    )  #
                )
                yield
            # on release
            if dragged:
                # dragged
                pass
            else:
                # just clicked
                pass


def add_transformed_fiducials_layer(*, ctx: AppContext) -> napari.layers.Points | None:
    logger.debug(f"Adding transformed fiducials layer")
    fiducials_key = "transformed_micro_fiducials"
    if fiducials_key in viewer().layers:
        logger.warning(f"Layer '{fiducials_key}' is already shown.")
        return None
    points: NamedPoints3D | Error = ctx.dadg.get(fiducials_key, soft=True)
    if isinstance(points, Error):
        raise RuntimeError(f"Error softly getting '{fiducials_key}' from DADG: {points.description}.")
    initial_points = points.data.flip(dims=(1,))
    logger.debug(f"Adding transformed fiducials layer '{fiducials_key}' to napari viewer")
    layer = viewer().add_points(  #
        initial_points.cpu().numpy(),  #
        ndim=3,  #
        size=3.0,  #
        symbol="ring",  #
        name=fiducials_key,  #
        features=pd.DataFrame([{"label": name} for name in points.names]),  #
        text={"string": "{label}", "size": 16}  #
    )
    layer.my_plugin = _TransformedFiducialsLayerManager(ctx=ctx, layer=layer)
    return layer
