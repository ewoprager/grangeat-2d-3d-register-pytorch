import os
import logging

os.environ["QT_API"] = "PyQt6"

from magicgui import widgets

from reg23_experiments.app.context import AppContext
from reg23_experiments.app.gui.moving_image_layer import add_moving_image_layer
from reg23_experiments.app.gui.fixed_image_layer import add_fixed_image_layer
from reg23_experiments.app.gui.electrode_layer import add_electrode_layer

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
            # Image 2d full
            show_image_2d_full_button = widgets.PushButton(label="Show full 2d image")
            show_image_2d_full_button.changed.connect(lambda _, name=key: self._on_show_image_2d_full_layer(name))
            # Fixed image
            show_fixed_image_button = widgets.PushButton(label="Show fixed image")
            show_fixed_image_button.changed.connect(lambda _, name=key: self._on_show_fixed_image_layer(name))
            # Moving image
            show_moving_image_button = widgets.PushButton(label="Show moving image")
            show_moving_image_button.changed.connect(lambda _, name=key: self._on_show_moving_image_layer(name))
            # Electrode points
            show_electrodes_button = widgets.PushButton(label="Show electrodes")
            show_electrodes_button.changed.connect(lambda _, name=key: self._on_show_electrode_layer(name))
            self._xray_container.append(widgets.Container(widgets=[  #
                show_image_2d_full_button,  #
                show_fixed_image_button,  #
                show_moving_image_button,  #
                show_electrodes_button  #
            ], label=key))

    def _on_show_image_2d_full_layer(self, xray_name: str) -> None:
        add_fixed_image_layer(ctx=self._ctx, dadg_key=f"{xray_name}__image_2d_full")

    def _on_show_fixed_image_layer(self, xray_name: str) -> None:
        add_fixed_image_layer(ctx=self._ctx, dadg_key=f"{xray_name}__fixed_image")

    def _on_show_moving_image_layer(self, xray_name: str) -> None:
        add_moving_image_layer(ctx=self._ctx, namespace=xray_name)

    def _on_show_electrode_layer(self, xray_name: str) -> None:
        add_electrode_layer(ctx=self._ctx, namespace=xray_name)
