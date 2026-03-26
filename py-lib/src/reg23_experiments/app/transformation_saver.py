import logging

from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.app.context import AppContext

__all__ = ["TransformationSaver"]

logger = logging.getLogger(__name__)


class TransformationSaver:
    def __init__(self, ctx: AppContext):
        self._ctx = ctx
        self._update_saved_transformation_names()
        self._ctx.state.observe(self._register_xray_choice_changed, names=["register_xray_choice"])
        self._ctx.state.observe(self._button_save_transformation, names=["button_save_transformation"])
        self._ctx.state.observe(self._button_load_transformation_of_name, names=["button_load_transformation_of_name"])
        self._ctx.state.observe(self._button_delete_transformation_of_name,
                                names=["button_delete_transformation_of_name"])

    @property
    def _xray_selected(self) -> bool:
        return self._ctx.state.register_xray_choice is not None

    @property
    def _uid_key(self) -> str:
        return f"{self._ctx.state.register_xray_choice}__xray_sop_instance_uid"

    @property
    def _c_t_key(self) -> str:
        return f"{self._ctx.state.register_xray_choice}__current_transformation"

    def _update_saved_transformation_names(self) -> None:
        if not self._xray_selected:
            self._ctx.state.saved_transformation_names = []
            return
        self._ctx.state.saved_transformation_names = self._ctx.transformation_save_manager.get_names(
            self._ctx.dadg.get(self._uid_key))

    def _register_xray_choice_changed(self, change) -> None:
        self._update_saved_transformation_names()

    def _button_save_transformation(self, change) -> None:
        if not change.new:
            return
        self._ctx.state.button_save_transformation = False
        #
        if not self._xray_selected:
            logger.warning("Cannot save transformation: no X-ray selected.")
            return
        if not self._ctx.state.text_input_transformation_name:
            logger.warning("Cannot save transformation: no name given.")
            return
        uid = self._ctx.dadg.get(self._uid_key)
        name = self._ctx.state.text_input_transformation_name
        err = self._ctx.transformation_save_manager.set(uid=uid, name=name,
                                                        transformation=self._ctx.dadg.get(self._c_t_key))
        if isinstance(err, Error):
            logger.error(f"Error saving transformation '{uid}; {name}' to save "
                         f"manager: {err.description}")
        self._update_saved_transformation_names()

    def _button_load_transformation_of_name(self, change) -> None:
        if change.new is None:
            return
        name = change.new
        self._ctx.state.button_load_transformation_of_name = None
        #
        if not self._xray_selected:
            logger.warning("Cannot load transformation: no X-ray selected.")
            return
        uid = self._ctx.dadg.get(self._uid_key)
        tr: Transformation | Error = self._ctx.transformation_save_manager.get_transformation(uid=uid, name=name)
        if isinstance(tr, Error):
            logger.error(f"Error loading transformation '{uid}; {name}' from save manager: {tr.description}")
            return
        device = self._ctx.dadg.get(self._c_t_key).rotation.device
        self._ctx.dadg.set(self._c_t_key, tr.to(device=device))

    def _button_delete_transformation_of_name(self, change) -> None:
        if change.new is None:
            return
        name = change.new
        self._ctx.state.button_delete_transformation_of_name = None
        #
        if not self._xray_selected:
            logger.warning("Cannot delete transformation: no X-ray selected.")
            return
        uid = self._ctx.dadg.get(self._uid_key)
        err = self._ctx.transformation_save_manager.remove(uid=uid, name=name)
        if isinstance(err, Error):
            logger.error(f"Error deleting transformation '{uid}; {name}' from save manager: {err.description}")
        self._update_saved_transformation_names()
