import logging
from typing import Callable

logger = logging.getLogger(__name__)

import torch
import napari
from magicgui import widgets

from registration.interface.registration_data import RegistrationData


class GrangeatWidget(widgets.Container):
    def __init__(self, *, moving_image_changed_signal, registration_data: RegistrationData,
                 render_moving_sinogram_callback: Callable[[], None],
                 fixed_image_crop_callback: Callable[[int, int], None]):
        super().__init__()
        moving_image_changed_signal.connect(self._on_moving_image_changed)
        self._registration_data = registration_data
        self._render_moving_sinogram_callback = render_moving_sinogram_callback

        self._regen_moving_sinogram_continuous: bool = False

        self._regen_moving_sinogram_once_button = widgets.PushButton(label="once")
        self._regen_moving_sinogram_once_button.changed.connect(self._on_regen_once)
        self._regen_moving_sinogram_continuous_check = widgets.CheckBox(label="continuous")
        self._regen_moving_sinogram_continuous_check.changed.connect(self._on_regen_continuous)
        self.append(widgets.Container(
            widgets=[self._regen_moving_sinogram_once_button, self._regen_moving_sinogram_continuous_check],
            layout="horizontal", label="Regen moving sinogram"))

        self._fixed_image_crop_callback = fixed_image_crop_callback
        self._ignore_sliders: bool = False
        height = self._registration_data.fixed_image.size()[0]
        self._bottom_slider = widgets.IntSlider(value=height, min=0, max=height, step=1, label="Crop bottom")
        self._top_slider = widgets.IntSlider(value=0, min=0, max=height, step=1, label="Crop top")
        self._bottom_slider.changed.connect(self._on_slider)
        self._top_slider.changed.connect(self._on_slider)
        self._bottom_slider.changed.connect(self._on_slider)
        self.append(self._top_slider)
        self.append(self._bottom_slider)

    def _on_regen_once(self) -> None:
        self._render_moving_sinogram_callback()

    def _on_regen_continuous(self, *args) -> None:
        self._regen_moving_sinogram_continuous = self._regen_moving_sinogram_continuous_check.get_value()
        self._on_regen_once()

    def _on_moving_image_changed(self) -> None:
        if self._regen_moving_sinogram_continuous:
            self._on_regen_once()

    def _on_slider(self, *args) -> None:
        if self._ignore_sliders:
            return
        self._ignore_sliders = True
        self._bottom_slider.min = self._top_slider.get_value() + 1
        self._top_slider.max = self._bottom_slider.get_value() - 1
        self._fixed_image_crop_callback(self._top_slider.get_value(), self._bottom_slider.get_value())
        self._ignore_sliders = False
