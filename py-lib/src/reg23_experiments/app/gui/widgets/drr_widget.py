import logging
import os

os.environ["QT_API"] = "PyQt6"

from magicgui import widgets

from reg23_experiments.app.gui.widgets.hastraits_widget import HasTraitsWidget
from reg23_experiments.app.context import AppContext

logger = logging.getLogger(__name__)

__all__ = ["DRRWidget"]


class DRRWidget(widgets.Container):
    def __init__(self, ctx: AppContext):
        super().__init__(widgets=[], layout="vertical", labels=False)

        self._ctx = ctx

        # -----
        # Parameters struct
        # -----
        self.append(widgets.Label(value="Parameters:"))
        self._traitlets_widget = HasTraitsWidget(self._ctx.state.drr_params)
        self._traitlets_widget.expanded = True
        self.append(self._traitlets_widget)

        # -----
        # Functionality
        # -----
        self.append(widgets.Label(value="Actions:"))

        self._create_fixed_image_button = widgets.PushButton(label="Create fixed image with name:")
        self._create_fixed_image_button.changed.connect(self._on_create_fixed_image)

        self._name_input = widgets.LineEdit(value=self._ctx.state.drr_name_input)
        self._name_input.changed.connect(self._on_name_input)

        self.append(widgets.Container(widgets=[  #
            self._create_fixed_image_button,  #
            self._name_input  #
        ], labels=False, layout="horizontal"))

    def _on_create_fixed_image(self, *args) -> None:
        self._ctx.state.button_create_drr = True

    def _on_name_input(self, *args) -> None:
        self._ctx.state.drr_name_input = self._name_input.get_value()
