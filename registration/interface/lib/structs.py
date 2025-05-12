from typing import Any
import logging

logger = logging.getLogger(__name__)

from magicgui import magicgui, widgets

class WidgetSelectData:
    def __init__(self, initial_choices: dict[str, Any] = {}):
        self.widget = widgets.Select(choices = [key for key in initial_choices])
        self.data = initial_choices

    def add_choice(self, name: str, data) -> None:
        self.widget.set_choice(name)
        self.data[name] = data

    def get_data(self, name: str) -> Any | None:
        if name in self.data:
            return self.data[name]
        return None

    def get_selected(self) -> str:
        return self.widget.current_choice

    def del_choices(self, names: list[str]) -> None:
        for name in names:
            if name not in self.data:
                continue
            self.widget.del_choice(name)
            self.data.pop(name)