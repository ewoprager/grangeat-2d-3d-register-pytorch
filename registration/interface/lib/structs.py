from typing import Any, NamedTuple
import logging

logger = logging.getLogger(__name__)

from magicgui import magicgui, widgets

class WidgetSelectData:
    def __init__(self, initial_choices: dict[str, Any] = {}):
        self._widget = widgets.Select(choices = [key for key in initial_choices])
        self._data = initial_choices

    @property
    def widget(self) -> widgets.Select:
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


class ViewParams(NamedTuple):
    translation_sensitivity: float
    rotation_sensitivity: float