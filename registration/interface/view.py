from typing import NamedTuple, Callable
import logging

logger = logging.getLogger(__name__)

import torch
from magicgui import widgets

from registration.interface.lib.structs import *


class ViewWidget(widgets.Container):
    def __init__(self, view_params_setter: Callable[[ViewParams], None]):
        super().__init__()
        self._view_params_setter = view_params_setter
        self._translation_sensitivity_slider = widgets.FloatSlider(value=0.06, min=0.005, max=0.5, step=0.005,
                                                                   label="Translation sensitivity")
        self._rotation_sensitivity_slider = widgets.FloatSlider(value=0.002, min=0.0005, max=0.05, step=0.0005,
                                                                label="Rotation sensitivity")
        self._translation_sensitivity_slider.changed.connect(self._update)
        self._rotation_sensitivity_slider.changed.connect(self._update)
        self.append(self._translation_sensitivity_slider)
        self.append(self._rotation_sensitivity_slider)

    def _update(self, *args) -> None:
        self._view_params_setter(ViewParams(translation_sensitivity=self._translation_sensitivity_slider.get_value(),
                                            rotation_sensitivity=self._rotation_sensitivity_slider.get_value()))
