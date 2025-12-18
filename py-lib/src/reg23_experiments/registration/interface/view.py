from typing import Callable
import logging

from magicgui import widgets

from reg23_experiments.registration.interface.lib.structs import ViewParams

__all__ = ["ViewWidget"]

logger = logging.getLogger(__name__)


class ViewWidget(widgets.Container):
    def __init__(self, view_params_setter: Callable[[ViewParams], None]):
        super().__init__()
        self._view_params_setter = view_params_setter

        self._fixed_image_mask_check = widgets.Checkbox(label="Render fixed image with mask")
        self._fixed_image_mask_check.changed.connect(self._update)
        self.append(self._fixed_image_mask_check)

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
                                            rotation_sensitivity=self._rotation_sensitivity_slider.get_value(),
                                            render_fixed_image_with_mask=self._fixed_image_mask_check.get_value()))
