import logging
from typing import Callable

logger = logging.getLogger(__name__)

import torch
import napari
from magicgui import widgets

from registration.lib.structs import *
from registration.interface.registration_data import RegistrationData
from registration.interface.transformations import TransformationWidget


class GrangeatWidget(widgets.Container):
    def __init__(self, moving_image_changed_signal, registration_data: RegistrationData,
                 transformation_widget: TransformationWidget, moving_sinogram_layer: napari.layers.Image):
        super().__init__()
        moving_image_changed_signal.connect(self._on_moving_image_changed)
        self._registration_data = registration_data
        self._transformation_widget = transformation_widget
        self._moving_sinogram_layer = moving_sinogram_layer

        self._regen_moving_sinogram_continuous: bool = False

        self._regen_moving_sinogram_once_button = widgets.PushButton(label="once")
        self._regen_moving_sinogram_once_button.changed.connect(self._on_regen_once)
        self._regen_moving_sinogram_continuous_check = widgets.CheckBox(label="continuous")
        self._regen_moving_sinogram_continuous_check.changed.connect(self._on_regen_continuous)
        self.append(widgets.Container(
            widgets=[self._regen_moving_sinogram_once_button, self._regen_moving_sinogram_continuous_check],
            layout="horizontal", label="Regen moving sinogram"))

    def _on_regen_once(self) -> None:
        self._moving_sinogram_layer.data = self._registration_data.resample_sinogram3d(
            self._transformation_widget.get_current_transformation()).cpu().numpy()
        print("regenerating!")

    def _on_regen_continuous(self, *args) -> None:
        self._regen_moving_sinogram_continuous = self._regen_moving_sinogram_continuous_check.get_value()

    def _on_moving_image_changed(self) -> None:
        if self._regen_moving_sinogram_continuous:
            self._on_regen_once()
