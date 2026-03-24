import os
import logging

os.environ["QT_API"] = "PyQt6"

import napari
from magicgui import widgets

from reg23_experiments.app.context import AppContext

__all__ = ["MainWidget"]

logger = logging.getLogger(__name__)


class MainWidget(widgets.Container):
    def __init__(self, ctx: AppContext):
        super().__init__(labels=False)
        self._ctx = ctx

        self._open_ct_file_button = widgets.PushButton(label="Open CT file")
        self._open_ct_file_button.changed.connect(self._on_open_ct_file)
        self._open_ct_dir_button = widgets.PushButton(label="Open CT directory")
        self._open_ct_dir_button.changed.connect(self._on_open_ct_dir)

        self.append(widgets.Container(widgets=[  #
            self._open_ct_file_button,  #
            self._open_ct_dir_button  #
        ], layout="horizontal"))

        self._open_xray_file_button = widgets.PushButton(label="Open X-ray file")
        self._open_xray_file_button.changed.connect(self._on_open_xray_file)
        self.append(self._open_xray_file_button)

    def _on_open_ct_file(self, *args) -> None:
        self._ctx.state.button_open_ct_file = True

    def _on_open_ct_dir(self, *args) -> None:
        self._ctx.state.button_open_ct_dir = True

    def _on_open_xray_file(self, *args) -> None:
        self._ctx.state.button_open_xray_file = True
