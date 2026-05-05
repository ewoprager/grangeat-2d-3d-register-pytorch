import logging
import os

os.environ["QT_API"] = "PyQt6"

from magicgui import widgets

from reg23_experiments.data.structs import Error
from reg23_app.context import AppContext

__all__ = ["FiducialsWidget"]

logger = logging.getLogger(__name__)


class FiducialsWidget(widgets.Container):
    def __init__(self, ctx: AppContext):
        super().__init__(widgets=[], layout='vertical', labels=False)

        self._ctx = ctx

        # -----
        # Register
        # -----
        self._register_button = widgets.PushButton(label="Register X-ray:")
        self._register_button.changed.connect(self._on_register)
        self._xray_select = widgets.ComboBox(choices=self._get_xray_choices)
        self._ctx.state.parameters.observe(self._xray_params_changed, names=["xray_parameters"])
        self._xray_select.changed.connect(self._on_xray_choice_changed)
        self._on_xray_choice_changed()
        self.append(widgets.Container(widgets=[  #
            self._register_button,  #
            self._xray_select,  #
        ], layout="horizontal", labels=False))

    def _on_register(self, *args) -> None:
        self._ctx.state.button_fiducial_register = True

    def _get_xray_choices(self, *args) -> list[str]:
        return list(self._ctx.state.parameters.xray_parameters.keys())

    def _xray_params_changed(self, change) -> None:
        self._xray_select.reset_choices()

    def _on_xray_choice_changed(self, *args) -> None:
        self._ctx.state.register_fiducial_xray_choice = self._xray_select.value
