import os
import logging

os.environ["QT_API"] = "PyQt6"

from magicgui import widgets

from reg23_experiments.app.context import AppContext
from reg23_experiments.app.gui.moving_image_layer import add_moving_image_layer

__all__ = ["MainWidget"]

logger = logging.getLogger(__name__)


class MainWidget(widgets.Container):
    def __init__(self, ctx: AppContext):
        super().__init__(labels=True)
        self._ctx = ctx

        self._ctx.state.parameters.observe(self._on_xray_parameters_change, names=["xray_parameters"])
        self._xray_container = widgets.Container(label="X-rays", labels=True)
        self.append(self._xray_container)
        self._on_xray_parameters_change()

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
