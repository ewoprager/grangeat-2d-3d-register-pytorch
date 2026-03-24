import os
import logging

os.environ["QT_API"] = "PyQt6"

import napari
from magicgui import widgets

from reg23_experiments.app.context import AppContext
from reg23_experiments.app.gui.moving_image_layer import add_moving_image_layer

__all__ = ["MainWidget"]

logger = logging.getLogger(__name__)


class MainWidget(widgets.Container):
    def __init__(self, ctx: AppContext):
        super().__init__(labels=True)
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

        self._ctx.state.parameters.observe(self._on_xray_parameters_change, names=["xray_parameters"])
        self._xray_container = widgets.Container(label="X-rays", labels=True)
        self.append(self._xray_container)
        self._on_xray_parameters_change()

    def _on_open_ct_file(self, *args) -> None:
        self._ctx.state.button_open_ct_file = True

    def _on_open_ct_dir(self, *args) -> None:
        self._ctx.state.button_open_ct_dir = True

    def _on_open_xray_file(self, *args) -> None:
        self._ctx.state.button_open_xray_file = True

    def _on_xray_parameters_change(self, *args) -> None:
        self._xray_container.clear()
        for key, value in self._ctx.state.parameters.xray_parameters.items():
            show_moving_image_button = widgets.PushButton(label="Show moving image")
            show_moving_image_button.changed.connect(lambda _, name=key: self._on_show_moving_image_layer(name))
            self._xray_container.append(widgets.Container(widgets=[  #
                show_moving_image_button  #
            ], label=key))

    def _on_show_moving_image_layer(self, xray_name: str) -> None:
        add_moving_image_layer(ctx=self._ctx, namespace=xray_name)
