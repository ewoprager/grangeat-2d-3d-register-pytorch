import logging
from typing import NamedTuple, Callable

import torch
import numpy as np
import scipy
from magicgui import widgets

from reg23_experiments.program.lib.structs import Error
from reg23_experiments.program import data_manager
from reg23_experiments.program.modules.interface import viewer
from reg23_experiments.registration.lib.structs import Transformation

__all__ = ["ViewParams", "ViewParamWidget", "MovingImageGUI"]

logger = logging.getLogger(__name__)


class ViewParams(NamedTuple):
    rotation_sensitivity: float
    translation_sensitivity: float


class ViewParamWidget(widgets.Container):
    def __init__(self, view_params_setter: Callable[[ViewParams], None]):
        super().__init__()
        self._view_params_setter = view_params_setter

        self.append(widgets.Label(label=None, value="Mouse drag sensitivity"))

        # Translation sensitivity
        self._translation_sensitivity_slider = widgets.FloatSlider(value=0.06, min=0.005, max=0.5, step=0.005,
                                                                   label="Translation")
        self._translation_sensitivity_slider.changed.connect(self._update)
        self.append(self._translation_sensitivity_slider)

        # Rotation sensitivity
        self._rotation_sensitivity_slider = widgets.FloatSlider(value=0.002, min=0.0005, max=0.05, step=0.0005,
                                                                label="Rotation")
        self._rotation_sensitivity_slider.changed.connect(self._update)
        self.append(self._rotation_sensitivity_slider)

    def _update(self, *args) -> None:
        self._view_params_setter(ViewParams(translation_sensitivity=self._translation_sensitivity_slider.get_value(),
                                            rotation_sensitivity=self._rotation_sensitivity_slider.get_value()))


class MovingImageGUI:
    def __init__(self):
        data_manager().add_callback("moving_image", "interface", self._set_callback)
        data_manager().set_evaluation_laziness("moving_image", lazily_evaluated=False)
        value = data_manager().get("moving_image", soft=True)
        if isinstance(value, Error):
            raise RuntimeError(f"Error softly getting 'moving_image' from DAG: {value.description}.")
        initial_image = value if isinstance(value, torch.Tensor) else torch.zeros((2, 2))
        self._layer = viewer().add_image(initial_image.cpu().numpy(), colormap="blue", blending="additive",
                                         interpolation2d="linear", name="DRR")
        self._layer.mouse_drag_callbacks.append(self._mouse_drag)
        viewer().bind_key("Control", self._on_ctrl_down)
        self._key_states = {"Ctrl": False}
        self._view_params = ViewParams(translation_sensitivity=0.06, rotation_sensitivity=0.002)
        self._view_widget = ViewParamWidget(self.set_view_params)
        viewer().window.add_dock_widget(self._view_widget, name="View options", area="left",
                                        menu=viewer().window.window_menu)

    def _set_callback(self, new_value: torch.Tensor) -> None:
        self._layer.data = new_value.cpu().numpy()

    def set_view_params(self, value: ViewParams) -> None:
        self._view_params = value

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
                rotvec=data_manager().get("current_transformation").rotation.cpu().numpy())
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True

                delta = self._view_params.rotation_sensitivity * (
                        np.array([event.position[1], -event.position[0]]) - drag_start)
                euler_angles = [delta[1], delta[0], 0.0]
                rot_euler = scipy.spatial.transform.Rotation.from_euler(seq="xyz", angles=euler_angles)
                rot_combined = rot_euler * rotation_start
                prev = data_manager().get("current_transformation")
                data_manager().set_data(  #
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
            translation_start = data_manager().get("current_transformation").translation[0:2].cpu()
            yield
            # on move
            while event.type == "mouse_move":
                dragged = True

                delta = self._view_params.translation_sensitivity * (torch.tensor(event.position) - drag_start).flip(
                    (0,))
                prev = data_manager().get("current_transformation")
                tr = prev.translation
                tr[0:2] = (translation_start + delta).to(device=tr.device)
                data_manager().set_data(  #
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
