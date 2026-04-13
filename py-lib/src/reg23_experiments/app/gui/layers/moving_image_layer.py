import logging
import weakref

import napari.layers
import numpy as np
import scipy
import torch

from reg23_experiments.app.context import AppContext
from reg23_experiments.app.gui.viewer_singleton import viewer
from reg23_experiments.data.structs import Error, Transformation

__all__ = ["add_moving_image_layer"]

logger = logging.getLogger(__name__)


class _MovingImageLayerManager:
    def __init__(self, *, ctx: AppContext, layer: napari.layers.Layer, namespace: str | None = None):
        self._ctx = ctx
        self._layer = weakref.ref(layer)
        self._namespace = namespace
        self._ctx.dadg.set_evaluation_laziness(
            "moving_image" if self._namespace is None else f"{self._namespace}__moving_image", lazily_evaluated=False)
        self._layer().mouse_drag_callbacks.append(self._mouse_drag)
        self._ctx.dadg.observe("moving_image" if self._namespace is None else f"{self._namespace}__moving_image",
                               "moving_image_manager", self._observer_callback)

    def __del__(self):
        self._ctx.dadg.set_evaluation_laziness(
            "moving_image" if self._namespace is None else f"{self._namespace}__moving_image", lazily_evaluated=True)

    def _observer_callback(self, new_value: torch.Tensor) -> None:
        self._layer().data = new_value.cpu().numpy()

    def _mouse_drag(self, layer, event):
        if event.button == 1 and self._ctx.input_manager.ctrl_pressed:  # Ctrl-left click drag
            # mouse down
            dragged = False
            drag_start = np.array([event.position[1], -event.position[0]])
            rotation_start = scipy.spatial.transform.Rotation.from_rotvec(rotvec=self._ctx.dadg.get(
                "current_transformation" if self._namespace is None else f"{self._namespace}__current_transformation").rotation.cpu().numpy())
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True

                delta = self._ctx.state.gui_settings.rotation_sensitivity * (
                        np.array([event.position[1], -event.position[0]]) - drag_start)
                euler_angles = [delta[1], delta[0], 0.0]
                rot_euler = scipy.spatial.transform.Rotation.from_euler(seq="xyz", angles=euler_angles)
                rot_combined = rot_euler * rotation_start
                prev = self._ctx.dadg.get(
                    "current_transformation" if self._namespace is None else f"{self._namespace}__current_transformation")
                self._ctx.dadg.set(  #
                    "current_transformation" if self._namespace is None else f"{self._namespace}__current_transformation",
                    #
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
            drag_start = torch.tensor(event.position)
            # rotation_start = scipy.spatial.transform.Rotation.from_rotvec(transformation.rotation.cpu().numpy())
            translation_start = self._ctx.dadg.get(
                "current_transformation" if self._namespace is None else f"{self._namespace}__current_transformation").translation[
                0:2].cpu()
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True

                delta = self._ctx.state.gui_settings.translation_sensitivity * (
                        torch.tensor(event.position) - drag_start).flip((0,))
                prev = self._ctx.dadg.get(
                    "current_transformation" if self._namespace is None else f"{self._namespace}__current_transformation")
                tr = prev.translation
                tr[0:2] = (translation_start + delta).to(device=tr.device)
                self._ctx.dadg.set(  #
                    "current_transformation" if self._namespace is None else f"{self._namespace}__current_transformation",
                    #
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


def add_moving_image_layer(*, ctx: AppContext, namespace: str | None = None) -> napari.layers.Layer | None:
    moving_image_key = "moving_image" if namespace is None else f"{namespace}__moving_image"
    if moving_image_key in viewer().layers:
        logger.warning(f"Layer '{moving_image_key}' is already shown.")
        return None
    ctx.dadg.set_evaluation_laziness(moving_image_key, lazily_evaluated=False)
    value = ctx.dadg.get(moving_image_key, soft=True)
    if isinstance(value, Error):
        raise RuntimeError(f"Error softly getting '{moving_image_key}' from DADG: {value.description}.")
    initial_image = value if isinstance(value, torch.Tensor) else torch.zeros((500, 500))
    layer = viewer().add_image(initial_image.cpu().numpy(), colormap="blue", blending="additive",
                               interpolation2d="linear", name=moving_image_key)
    layer.my_plugin = _MovingImageLayerManager(ctx=ctx, layer=layer, namespace=namespace)
    return layer
