import logging

from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.data.transformation_save_data import TransformationSaveManager
from reg23_experiments.app.state import AppState

__all__ = ["SaveDataManager"]

logger = logging.getLogger(__name__)


class SaveDataManager:
    def __init__(self, app_state: AppState):
        self._app_state = app_state
        self._transformation_save_manager = TransformationSaveManager(self._app_state.transformation_save_directory)
        self._update_saved_transformation_names()
        self._app_state.observe(self._button_save_transformation, names=["button_save_transformation"])
        self._app_state.observe(self._button_load_transformation_of_name, names=["button_load_transformation_of_name"])
        self._app_state.observe(self._button_delete_transformation_of_name,
                                names=["button_delete_transformation_of_name"])

    def _update_saved_transformation_names(self) -> None:
        self._app_state.saved_transformation_names = self._transformation_save_manager.get_names()

    def _button_save_transformation(self, change) -> None:
        if not change.new:
            return
        self._app_state.button_save_transformation = False
        if not self._app_state.text_input_transformation_name:
            logger.warning("Cannot save transformation; no name given.")
            return
        err = self._transformation_save_manager.set(self._app_state.text_input_transformation_name,
                                                    self._app_state.dadg.get("current_transformation"))
        if isinstance(err, Error):
            logger.error(
                f"Error saving transformation with name '{self._app_state.text_input_transformation_name}' to save "
                f"manager: {err.description}")
        self._update_saved_transformation_names()
        self._app_state.button_save_transformation = False

    def _button_load_transformation_of_name(self, change) -> None:
        if change.new is None:
            return
        name = self._app_state.button_load_transformation_of_name
        self._app_state.button_load_transformation_of_name = None
        tr: Transformation | Error = self._transformation_save_manager.get_transformation(name)
        if isinstance(tr, Error):
            logger.error(f"Error loading transformation '{name}' from save manager: {tr.description}")
            return
        device = self._app_state.dadg.get("current_transformation").rotation.device
        self._app_state.dadg.set("current_transformation", tr.to(device=device))

    def _button_delete_transformation_of_name(self, change) -> None:
        if change.new is None:
            return
        name = self._app_state.button_delete_transformation_of_name
        self._app_state.button_delete_transformation_of_name = None
        err = self._transformation_save_manager.remove(name)
        if isinstance(err, Error):
            logger.error(f"Error deleting transformation '{name}' from save manager: {err.description}")
        self._update_saved_transformation_names()
