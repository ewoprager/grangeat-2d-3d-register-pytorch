from typing import Any, NamedTuple, Callable
import pickle
from datetime import datetime
import logging

from reg23_experiments.registration.lib.structs import Transformation

import torch
import pathlib
from magicgui import widgets

__all__ = ["Target", "Cropping", "WidgetSelectData", "WidgetManageSaved", "ViewParams", "SavedXRayParams"]

logger = logging.getLogger(__name__)


class Target(NamedTuple):
    xray_path: str | None  # None indicates DRR
    flipped: bool = False  # horizontal flipping of the image


class Cropping(NamedTuple):
    right: int
    top: int
    left: int
    bottom: int

    @staticmethod
    def zero(image_size: torch.Size) -> 'Cropping':
        return Cropping(right=image_size[1], top=0, left=0, bottom=image_size[0])

    def get_centre_offset(self, full_size: torch.Size) -> torch.Tensor:
        top_left = torch.tensor([self.left, self.top], dtype=torch.float64)
        size = torch.tensor([self.right - self.left, self.bottom - self.top], dtype=torch.float64)
        return top_left + 0.5 * size - (0.5 * torch.tensor(full_size, dtype=torch.float64).flip(dims=(0,)))

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[self.top:self.bottom, self.left:self.right]

    def to_downsample_level(self, downsample_level: int, *, image_size: torch.Size) -> 'Cropping':
        downsample_factor = int(2 ** downsample_level)
        return Cropping(top=self.top // downsample_factor,
                        bottom=min(self.bottom // downsample_factor, image_size[0] - 1),
                        left=self.left // downsample_factor,
                        right=min(self.right // downsample_factor, image_size[1] - 1))


# class HyperParameters(NamedTuple):
#     cropping: Cropping
#     source_offset: torch.Tensor  # size = (2,), dtype = torch.float64
#     downsample_level: int  # downsample factor is 2^downsample_level


class WidgetSelectData:
    def __init__(self, *, widget_type: type, initial_choices: dict[str, Any] = {}, **kwargs):
        self._widget = widget_type(choices=[key for key in initial_choices], **kwargs)
        self._data = initial_choices

    @property
    def widget(self) -> Any:
        return self._widget

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def add_choice(self, name: str, data) -> None:
        self._widget.set_choice(name)
        self._data[name] = data

    def get_data(self, name: str) -> Any | None:
        if self.name_exists(name):
            return self._data[name]
        return None

    def name_exists(self, name: str) -> bool:
        return name in self._data

    def get_selected(self) -> str:
        return self._widget.current_choice

    def del_choices(self, names: list[str]) -> None:
        for name in names:
            if name not in self._data:
                continue
            self._widget.del_choice(name)
            self._data.pop(name)


class WidgetManageSaved(widgets.Container):
    def __init__(self, *, initial_choices: dict[str, Any] = {}, DataType: type,
                 load_from_file: str | pathlib.Path | None, get_current_callback: Callable[[], Any],
                 set_callback: Callable[[Any], None]):
        super().__init__()

        self._get_current_callback = get_current_callback
        self._set_callback = set_callback

        self._save_button = widgets.PushButton(label="Save new")
        self._save_button.changed.connect(self._on_save)

        self._set_to_saved_button = widgets.PushButton(label="Load selected")
        self._set_to_saved_button.changed.connect(self._on_set_to_saved)

        self._del_button = widgets.PushButton(label="Delete selected")
        self._del_button.changed.connect(self._on_del)

        self.append(widgets.Container(widgets=[self._save_button, self._set_to_saved_button, self._del_button],
                                      layout="horizontal"))

        if load_from_file is not None:
            if pathlib.Path(load_from_file).is_file():
                with open(load_from_file, "rb") as file:
                    loaded = pickle.load(file)
                    valid = isinstance(loaded, dict)
                    if valid:
                        for k, v in loaded.items():
                            valid = valid and isinstance(k, str) and isinstance(v, DataType)
                    if valid:
                        initial_choices = initial_choices | loaded
                        logger.info("Saved data loaded from '{}'".format(str(load_from_file)))
                    else:
                        logger.warning("Invalid saved data at '{}'".format(str(load_from_file)))
            else:
                logger.warning("Save file '{}' doesn't exist.".format(str(load_from_file)))
        self._select_saved_widget = WidgetSelectData(widget_type=widgets.Select, initial_choices=initial_choices,
                                                     label="Saved:")
        self.append(self._select_saved_widget.widget)

        self._rename_input = widgets.LineEdit(value=datetime.now().strftime("%Y-%m-%d"))
        self._rename_widget = widgets.PushButton(label="Rename selected")
        self._rename_widget.changed.connect(self._on_rename)
        self.append(
            widgets.Container(widgets=[self._rename_input, self._rename_widget], labels=False, layout="horizontal",
                              label="Rename to"))

    def add_value(self, name: str, data: Any) -> None:
        while self._select_saved_widget.name_exists(name):
            name = "{} (1)".format(name)
        self._select_saved_widget.add_choice(name, data)

    def save_to_file(self, path: str | pathlib.Path) -> None:
        with open(path, "wb") as file:
            pickle.dump(self._select_saved_widget.data, file)
            logger.info("Values saved to '{}'".format(str(path)))

    def _on_save(self) -> None:
        new_name = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        self._select_saved_widget.add_choice(new_name, self._get_current_callback())

    def _on_set_to_saved(self) -> None:
        current = self._select_saved_widget.get_selected()
        if len(current) == 1:
            # self.set_current_transformation(self._select_saved_widget.get_data(current[0]))
            self._set_callback(self._select_saved_widget.get_data(current[0]))
        elif len(current) == 0:
            logger.warning("No value selected.")
        else:
            logger.warning("Multiple values selected.")

    def _on_del(self) -> None:
        current = self._select_saved_widget.get_selected()  # this actually returns a list of strings
        if len(current) == 0:
            logger.warning("No value selected.")
            return
        self._select_saved_widget.del_choices(current)

    def _on_rename(self) -> None:
        new_name = self._rename_input.get_value()
        if self._select_saved_widget.name_exists(new_name):
            logger.warning("Value already exists with name '{}'.".format(new_name))
            return
        current = self._select_saved_widget.get_selected()
        if len(current) == 1:
            data = self._select_saved_widget.get_data(current[0])
            self._select_saved_widget.del_choices(current)
            self._select_saved_widget.add_choice(new_name, data)
        elif len(current) == 0:
            logger.warning("No value selected.")
        else:
            logger.warning("Multiple values selected.")


class ViewParams(NamedTuple):
    translation_sensitivity: float
    rotation_sensitivity: float
    render_fixed_image_with_mask: bool


class SavedXRayParams(NamedTuple):
    transformation: Transformation
    cropping: Cropping
    source_offset: torch.Tensor
    flipped: bool
