import logging

from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.app.context import AppContext

__all__ = ["TransformationSaver"]

logger = logging.getLogger(__name__)


class TransformationSaver:
    def __init__(self, ctx: AppContext):
        self._ctx = ctx
        self._update_saved_transformation_names()
        self._ctx.state.observe(self._button_save_transformation, names=["button_save_transformation"])
        self._ctx.state.observe(self._button_load_transformation_of_name, names=["button_load_transformation_of_name"])
        self._ctx.state.observe(self._button_delete_transformation_of_name,
                                names=["button_delete_transformation_of_name"])

    def _update_saved_transformation_names(self) -> None:
        self._ctx.state.saved_transformation_names = self._ctx.transformation_save_manager.get_names(
            self._ctx.dadg.get("xray_sop_instance_uid"))

    def _button_save_transformation(self, change) -> None:
        if not change.new:
            return
        self._ctx.state.button_save_transformation = False
        if not self._ctx.state.text_input_transformation_name:
            logger.warning("Cannot save transformation; no name given.")
            return
        uid = self._ctx.dadg.get("xray_sop_instance_uid")
        name = self._ctx.state.text_input_transformation_name
        err = self._ctx.transformation_save_manager.set(uid=uid, name=name,
                                                        transformation=self._ctx.dadg.get("current_transformation"))
        if isinstance(err, Error):
            logger.error(f"Error saving transformation '{uid}; {name}' to save "
                         f"manager: {err.description}")
        self._update_saved_transformation_names()
        self._ctx.state.button_save_transformation = False

    def _button_load_transformation_of_name(self, change) -> None:
        if change.new is None:
            return
        uid = self._ctx.dadg.get("xray_sop_instance_uid")
        name = self._ctx.state.button_load_transformation_of_name
        self._ctx.state.button_load_transformation_of_name = None
        tr: Transformation | Error = self._ctx.transformation_save_manager.get_transformation(uid=uid, name=name)
        if isinstance(tr, Error):
            logger.error(f"Error loading transformation '{uid}; {name}' from save manager: {tr.description}")
            return
        device = self._ctx.dadg.get("current_transformation").rotation.device
        self._ctx.dadg.set("current_transformation", tr.to(device=device))

    def _button_delete_transformation_of_name(self, change) -> None:
        if change.new is None:
            return
        uid = self._ctx.dadg.get("xray_sop_instance_uid")
        name = self._ctx.state.button_delete_transformation_of_name
        self._ctx.state.button_delete_transformation_of_name = None
        err = self._ctx.transformation_save_manager.remove(uid=uid, name=name)
        if isinstance(err, Error):
            logger.error(f"Error deleting transformation '{uid}; {name}' from save manager: {err.description}")
        self._update_saved_transformation_names()
