import logging

from reg23_experiments.app.state import AppState

__all__ = ["Controller"]

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, app_state: AppState):
        self._app_state = app_state

        self._app_state.observe(self._button_evaluate_once, names=["button_evaluate_once"])

    def _button_evaluate_once(self, change) -> None:
        if change.new != True:
            return
        logger.info("Button evaluate once pressed.")
        self._app_state.button_evaluate_once = False
