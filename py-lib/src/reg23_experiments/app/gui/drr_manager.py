import logging
import pathlib

from magicgui.widgets import request_values
from traitlets import TraitError

from reg23_experiments.app.gui.viewer_singleton import viewer
from reg23_experiments.app.state import AppState
from reg23_experiments.experiments.parameters import XrayParameters

__all__ = ["DRRManager"]

logger = logging.getLogger(__name__)


class DRRManager:
    """
    No widgets

    Reads from and write to the state only
    """

    def __init__(self, state: AppState):
        self._state = state

        self._state.observe(self._button_create_drr, names=["button_create_drr"])

    def _button_create_drr(self, change) -> None:
        if not change.new:
            return
        self._state.button_create_drr = False

        if self._state.drr_name_input in self._state.parameters.xray_parameters:
            logger.error(f"Can't create DRR with name '{self._state.drr_name_input}' as this name is already in use.")
            return

        # ToDo