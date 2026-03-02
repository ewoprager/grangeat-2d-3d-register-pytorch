import logging

from magicgui import widgets

from reg23_experiments.app.gui.helpers import TraitletsWidget
from reg23_experiments.experiments.parameters import Parameters
from reg23_experiments.app.state import AppState
from reg23_experiments.ops.data_manager import args_from_dadg
from reg23_experiments.ops.geometry import get_crop_nonzero_drr, get_crop_full_depth_drr

__all__ = ["ParametersWidget"]

logger = logging.getLogger(__name__)


class ParametersWidget(widgets.Container):
    def __init__(self, app_state: AppState, parameters: Parameters):
        super().__init__(widgets=[], layout='vertical', labels=False)

        self._app_state = app_state

        self.append(widgets.Label(value="Values:"))
        self.append(TraitletsWidget(parameters))

        self.append(widgets.Label(value="Modifiers:"))

        self._crop_nonzero_drr_button = widgets.PushButton(label="Crop to nonzero drr")
        self._crop_nonzero_drr_button.changed.connect(self._on_crop_nonzero_drr)
        self.append(self._crop_nonzero_drr_button)

        self._crop_full_depth_drr_button = widgets.PushButton(label="Crop to full depth drr")
        self._crop_full_depth_drr_button.changed.connect(self._on_crop_full_depth_drr)
        self.append(self._crop_full_depth_drr_button)

    def _on_crop_nonzero_drr(self, *args) -> None:
        self._app_state.parameters.cropping = "fixed"
        self._app_state.parameters.cropping_value = args_from_dadg(dadg=self._app_state.dadg)(get_crop_nonzero_drr)()

    def _on_crop_full_depth_drr(self, *args) -> None:
        self._app_state.parameters.cropping = "fixed"
        self._app_state.parameters.cropping_value = args_from_dadg(dadg=self._app_state.dadg)(get_crop_full_depth_drr)()
