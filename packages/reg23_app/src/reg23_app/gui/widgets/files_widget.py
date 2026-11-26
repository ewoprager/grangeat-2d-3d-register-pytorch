import logging
import os

os.environ["QT_API"] = "PyQt6"

from magicgui import widgets

from reg23_app.context import AppContext

__all__ = ["FilesWidget"]

logger = logging.getLogger(__name__)


class FilesWidget(widgets.Container):
    def __init__(self, ctx: AppContext, can_load_more_images: bool = True):
        super().__init__(widgets=[], layout='vertical', labels=False)

        self._ctx = ctx

        if can_load_more_images:
            # CT file opening
            self._open_ct_file_button = widgets.PushButton(label="Open CT file")
            self._open_ct_file_button.changed.connect(self._on_open_ct_file)
            self._open_ct_dir_button = widgets.PushButton(label="Open CT directory")
            self._open_ct_dir_button.changed.connect(self._on_open_ct_dir)

            self.append(widgets.Container(widgets=[  #
                self._open_ct_file_button,  #
                self._open_ct_dir_button  #
            ], layout="horizontal", labels=False))

            # microCT opening
            self._open_micro_ct_button = widgets.PushButton(label="Open microCT")
            self._open_micro_ct_button.changed.connect(self._on_open_micro_ct)
            self.append(self._open_micro_ct_button)

            # X-ray file opening
            self._open_xray_file_button = widgets.PushButton(label="Open X-ray file")
            self._open_xray_file_button.changed.connect(self._on_open_xray_file)
            self.append(self._open_xray_file_button)

            # X-ray file unloading
            self._unload_xray_file_button = widgets.PushButton(label="Unload X-ray file: ")
            self._unload_xray_file_button.changed.connect(self._on_unload_xray_file)
            self._unload_xray_select = widgets.ComboBox(choices=self._get_xray_choices)
            self._ctx.state.parameters.observe(self._xray_params_changed, names=["xray_parameters"])
            self._unload_xray_select.changed.connect(self._on_unload_xray_choice_changed)
            self._on_unload_xray_choice_changed()
            self.append(widgets.Container(widgets=[  #
                self._unload_xray_file_button,  #
                self._unload_xray_select  #
            ], layout="horizontal", labels=False))

    def _on_open_ct_file(self, *args) -> None:
        self._ctx.state.button_open_ct_file = True

    def _on_open_ct_dir(self, *args) -> None:
        self._ctx.state.button_open_ct_dir = True

    def _on_open_micro_ct(self, *args) -> None:
        self._ctx.state.button_open_micro_ct = True

    def _on_open_xray_file(self, *args) -> None:
        self._ctx.state.button_open_xray_file = True

    def _on_unload_xray_file(self, *args) -> None:
        self._ctx.state.button_unload_xray_file = True

    def _xray_params_changed(self, change) -> None:
        self._unload_xray_select.reset_choices()

    def _get_xray_choices(self, *args) -> list[str]:
        return list(self._ctx.state.parameters.xray_parameters.keys())

    def _on_unload_xray_choice_changed(self, *args) -> None:
        self._ctx.state.unload_xray_choice = self._unload_xray_select.value
