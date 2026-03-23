import logging
import weakref

import napari.layers
import torch
import numpy as np
import scipy

from reg23_experiments.data.structs import Error
from reg23_experiments.app.gui.viewer_singleton import viewer
from reg23_experiments.data.structs import Transformation
from reg23_experiments.app.state import AppState

__all__ = ["add_moving_image_layer"]

logger = logging.getLogger(__name__)


class _MovingImageManager:
    def __init__(self, *, layer: napari.layers.Layer, app_state: AppState, dadg_key: str):
        self._app_state = app_state
        self._app_state.dadg.set_evaluation_laziness(dadg_key, lazily_evaluated=False)
        self._layer = weakref.ref(layer)
        self._layer().mouse_drag_callbacks.append(self._mouse_drag)
        viewer().bind_key("Control", self._on_ctrl_down)
        self._key_states = {"Ctrl": False}
        self._app_state.dadg.observe(dadg_key, "interface", self._set_callback)

    def _set_callback(self, new_value: torch.Tensor) -> None:
        self._layer().data = new_value.cpu().numpy()

    def _on_ctrl_down(self, _):
        self._key_states["Ctrl"] = True
        yield
        self._key_states["Ctrl"] = False

    def _mouse_drag(self, layer, event):
        if event.button == 1 and self._key_states["Ctrl"]:  # Ctrl-left click drag
            # mouse down
            dragged = False
            drag_start = np.array([event.position[1], -event.position[0]])
            rotation_start = scipy.spatial.transform.Rotation.from_rotvec(
                rotvec=self._app_state.dadg.get("current_transformation").rotation.cpu().numpy())
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True

                delta = self._app_state.gui_settings.rotation_sensitivity * (
                        np.array([event.position[1], -event.position[0]]) - drag_start)
                euler_angles = [delta[1], delta[0], 0.0]
                rot_euler = scipy.spatial.transform.Rotation.from_euler(seq="xyz", angles=euler_angles)
                rot_combined = rot_euler * rotation_start
                prev = self._app_state.dadg.get("current_transformation")
                self._app_state.dadg.set(  #
                    "current_transformation",  #
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
        elif event.button == 2 and self._key_states["Ctrl"]:  # Ctrl-right click drag
            # mouse down
            dragged = False
            drag_start = torch.tensor(event.position)
            # rotation_start = scipy.spatial.transform.Rotation.from_rotvec(transformation.rotation.cpu().numpy())
            translation_start = self._app_state.dadg.get("current_transformation").translation[0:2].cpu()
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True

                delta = self._app_state.gui_settings.translation_sensitivity * (
                        torch.tensor(event.position) - drag_start).flip((0,))
                prev = self._app_state.dadg.get("current_transformation")
                tr = prev.translation
                tr[0:2] = (translation_start + delta).to(device=tr.device)
                self._app_state.dadg.set(  #
                    "current_transformation",  #
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


def add_moving_image_layer(app_state: AppState, namespace: str | None = None) -> napari.layers.Layer:
    moving_image_key = "moving_image" if namespace is None else f"{namespace}__moving_image"
    app_state.dadg.set_evaluation_laziness(moving_image_key, lazily_evaluated=False)
    value = app_state.dadg.get(moving_image_key, soft=True)
    if isinstance(value, Error):
        raise RuntimeError(f"Error softly getting '{moving_image_key}' from DADG: {value.description}.")
    initial_image = value if isinstance(value, torch.Tensor) else torch.zeros((500, 500))
    layer = viewer().add_image(initial_image.cpu().numpy(), colormap="blue", blending="additive",
                               interpolation2d="linear", name=moving_image_key)
    layer.my_plugin = _MovingImageManager(layer=layer, app_state=app_state, dadg_key=moving_image_key)
    return layer
