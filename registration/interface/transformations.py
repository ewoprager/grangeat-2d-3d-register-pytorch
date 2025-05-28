import pathlib
from typing import NamedTuple, Callable
from datetime import datetime
import pickle
import logging

from magicgui.types import Undefined

logger = logging.getLogger(__name__)

import torch
import napari
from magicgui import widgets
from qtpy.QtWidgets import QApplication

from registration.lib.structs import Transformation
from registration.interface.lib.structs import *


# class SavedTransformation(NamedTuple):
#     name: str
#     value: Transformation


class TransformationWidget(widgets.Container):
    def __init__(self, *, initial_transformation: Transformation, refresh_render_function: Callable[[], None],
                 save_path: pathlib.Path):
        super().__init__(labels=False)
        self._current_transformation = initial_transformation
        self._refresh_render_function = refresh_render_function
        self._save_path = save_path
        self._suppress_callbacks: bool = False

        self._translation_widgets: list[widgets.FloatSpinBox] = []
        for i, dim in enumerate(['X', 'Y', 'Z']):
            self._translation_widgets.append(
                widgets.FloatSpinBox(value=self._current_transformation.translation[i].item(), step=0.2,
                                     label="{} translation".format(dim), min=-1e8, max=1e8))
            self._translation_widgets[-1].changed.connect(self._update_transformations)

        self._rotation_widgets: list[widgets.FloatSpinBox] = []
        for i, dim in enumerate(['X', 'Y', 'Z']):
            self._rotation_widgets.append(
                widgets.FloatSpinBox(value=self._current_transformation.rotation[i].item(), step=0.01,
                                     label="{} rotation".format(dim), min=-1e8, max=1e8))
            self._rotation_widgets[-1].changed.connect(self._update_transformations)

        self.append(widgets.Container(widgets=self._translation_widgets + self._rotation_widgets, layout="vertical"))

        initial_choices = {"initial": self._current_transformation}
        self._save_widget = WidgetManageSaved(initial_choices=initial_choices, DataType=Transformation,
                                              load_from_file=self._save_path,
                                              get_current_callback=self.get_current_transformation,
                                              set_callback=self.set_current_transformation)
        self.append(self._save_widget)

        QApplication.instance().aboutToQuit.connect(self._on_exit)

    def get_current_transformation(self) -> Transformation:
        return self._current_transformation

    def set_current_transformation(self, new_value: Transformation) -> None:
        self._current_transformation = new_value.to(device=self._current_transformation.device())
        self._suppress_callbacks = True
        for i in range(3):
            self._translation_widgets[i].set_value(self._current_transformation.translation[i].item())
        for i in range(3):
            self._rotation_widgets[i].set_value(self._current_transformation.rotation[i].item())
        self._suppress_callbacks = False
        self._refresh_render_function()

    def save_transformation(self, transformation: Transformation, name: str) -> None:
        self._save_widget.add_value(name, transformation)

    def _update_transformations(self, *args) -> None:
        if self._suppress_callbacks:
            return
        self.set_current_transformation(
            Transformation(translation=torch.tensor([slider.get_value() for slider in self._translation_widgets]),
                           rotation=torch.tensor([slider.get_value() for slider in self._rotation_widgets])))

    def _on_exit(self) -> None:
        self._save_widget.save_to_file(self._save_path)
