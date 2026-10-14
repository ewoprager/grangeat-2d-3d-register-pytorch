from typing import Sequence

from magicgui import widgets

__all__ = ["ExpandableWidget"]


class ExpandableWidget(widgets.Container):
    def __init__(self, contained: Sequence[widgets.Widget], **widget_kwargs):
        if "layout" not in widget_kwargs:
            widget_kwargs["layout"] = "vertical"
        if "labels" not in widget_kwargs:
            widget_kwargs["labels"] = False
        super().__init__(widgets=[], **widget_kwargs)

        self._expand_toggle = widgets.CheckBox(value=False, text="expand")
        self._expand_toggle.changed.connect(self._expand_toggled)
        self.append(self._expand_toggle)
        self._inner_container = widgets.Container(widgets=contained, layout="vertical", labels=True)
        self.append(self._inner_container)

        self._expand_toggled()

    @property
    def expanded(self) -> bool:
        return self._expand_toggle.value

    @expanded.setter
    def expanded(self, value) -> None:
        self._expand_toggle.value = value

    def _expand_toggled(self, *args) -> None:
        if self._expand_toggle.value:
            if self._inner_container not in self:
                self.append(self._inner_container)
        elif self._inner_container in self:
            self.remove(self._inner_container)
