import pathlib
from typing import NamedTuple, Callable
from datetime import datetime
import pickle
import logging

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
    def __init__(self, *, initial_transformation: Transformation,
                 refresh_render_function: Callable[[Transformation], None], save_path: pathlib.Path):
        super().__init__()
        self._current_transformation = initial_transformation
        self._refresh_render_function = refresh_render_function
        self._save_path = save_path
        self._suppress_callbacks: bool = False

        self._translation_sliders: list[widgets.FloatSlider] = []
        for i, dim in enumerate(['X', 'Y', 'Z']):
            self._translation_sliders.append(
                widgets.FloatSlider(value=self._current_transformation.translation[i].item(), min=-300.0, max=300.0,
                                    step=0.2, label="{} translation".format(dim)))
            self._translation_sliders[-1].changed.connect(self._update_transformations)
            self.append(self._translation_sliders[-1])

        self._rotation_sliders: list[widgets.FloatSlider] = []
        for i, dim in enumerate(['X', 'Y', 'Z']):
            self._rotation_sliders.append(
                widgets.FloatSlider(value=self._current_transformation.rotation[i].item(), min=-10.0, max=10.0,
                                    step=0.01, label="{} rotation".format(dim)))
            self._rotation_sliders[-1].changed.connect(self._update_transformations)
            self.append(self._rotation_sliders[-1])

        self._save_button = widgets.PushButton(label="Save new")
        self._save_button.changed.connect(self._on_save)

        self._set_to_saved_button = widgets.PushButton(label="Load selected")
        self._set_to_saved_button.changed.connect(self._on_set_to_saved)

        self._del_button = widgets.PushButton(label="Delete selected")
        self._del_button.changed.connect(self._on_del)

        self.append(widgets.Container(widgets=[self._save_button, self._set_to_saved_button, self._del_button],
                                      layout="horizontal"))

        initial_choices = {"initial": self._current_transformation}
        if self._save_path.is_file():
            with open(self._save_path, "rb") as file:
                loaded = pickle.load(file)
                valid = isinstance(loaded, dict)
                if valid:
                    for k, v in loaded.items():
                        valid = valid and isinstance(k, str) and isinstance(v, Transformation)
                if valid:
                    initial_choices = initial_choices | loaded
                    logger.info("Saved transformation data loaded from '{}'".format(str(self._save_path)))
                else:
                    logger.warning("Invalid saved transformation data at '{}'".format(str(self._save_path)))
        else:
            logger.warning("Transformation save file '{}' doesn't exist.".format(str(self._save_path)))
        self._select_saved_widget = WidgetSelectData(initial_choices=initial_choices)
        self.append(self._select_saved_widget.widget)

        QApplication.instance().aboutToQuit.connect(self._on_exit)

    def get_current_transformation(self) -> Transformation:
        return self._current_transformation

    def set_current_transformation(self, new_value: Transformation) -> None:
        self._current_transformation = new_value.to(device=self._current_transformation.device())
        self._suppress_callbacks = True
        for i in range(3):
            self._translation_sliders[i].set_value(self._current_transformation.translation[i].item())
        for i in range(3):
            self._rotation_sliders[i].set_value(self._current_transformation.rotation[i].item())
        self._suppress_callbacks = False
        self._refresh_render_function(self._current_transformation)

    def save_transformation(self, transformation: Transformation, name: str) -> None:
        while self._select_saved_widget.name_exists(name):
            name = "{} (1)".format(name)
        self._select_saved_widget.add_choice(name, transformation)

    def _update_transformations(self, *args) -> None:
        if self._suppress_callbacks:
            return
        self.set_current_transformation(
            Transformation(translation=torch.tensor([slider.get_value() for slider in self._translation_sliders]),
                           rotation=torch.tensor([slider.get_value() for slider in self._rotation_sliders])))

    def _on_exit(self) -> None:
        with open(self._save_path, "wb") as file:
            pickle.dump(self._select_saved_widget.data, file)
            logger.info("Transformation data saved to '{}'".format(str(self._save_path)))

    def _on_save(self) -> None:
        if self._suppress_callbacks:
            return
        new_name = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        self._select_saved_widget.add_choice(new_name, self._current_transformation)

    def _on_set_to_saved(self) -> None:
        if self._suppress_callbacks:
            return
        current = self._select_saved_widget.get_selected()
        if len(current) == 1:
            self.set_current_transformation(self._select_saved_widget.get_data(current[0]))
        elif len(current) == 0:
            logger.warning("No transformation selected.")
        else:
            logger.warning("Multiple transformations selected.")

    def _on_del(self) -> None:
        if self._suppress_callbacks:
            return
        current = self._select_saved_widget.get_selected()  # this actually returns a list of strings
        if len(current) == 0:
            logger.warning("No transformation selected.")
            return
        self._select_saved_widget.del_choices(current)
