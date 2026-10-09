import logging

from reg23_app.context import AppContext
from reg23_experiments.data.structs import Error, Transformation

__all__ = ["TransformationSaver"]

logger = logging.getLogger(__name__)


class TransformationSaver:
    """
    No GUI / widgets

    Reads from and writes to the state
    """

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
    def _c_t_key(self) -> str:
        return f"{self._ctx.state.register_xray_choice}__current_transformation"

    def _get_ct_xray_uids(self) -> tuple[str, str] | Error:
        xray_uid_key = f"{self._ctx.state.register_xray_choice}__xray_sop_instance_uid"
        ct_uid_key = "ct_series_uid"

        ct_uid: str | Error = self._ctx.dadg.get(ct_uid_key)
        if isinstance(ct_uid, Error):
            return ct_uid
        xray_uid: str | Error = self._ctx.dadg.get(xray_uid_key)
        if isinstance(xray_uid, Error):
            return xray_uid

        return ct_uid, xray_uid

    def _update_saved_transformation_names(self) -> None:
        if not self._xray_selected:
            self._ctx.state.saved_transformation_names = []
            return
        res: tuple[str, str] | Error = self._get_ct_xray_uids()
        if isinstance(res, Error):
            logger.error(f"Failed to update saved transformation names: {res.description}")
            return
        ct_uid, xray_uid = res
        self._ctx.state.saved_transformation_names = self._ctx.transformation_save_manager.get_list_of_names(  #
            source_uid=ct_uid,  #
            destination_uid=xray_uid,  #
        )

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
        res: tuple[str, str] | Error = self._get_ct_xray_uids()
        if isinstance(res, Error):
            logger.error(f"Failed to save transformation: {res.description}")
            return
        ct_uid, xray_uid = res
        curr_tr: Transformation | Error = self._ctx.dadg.get(self._c_t_key)
        if isinstance(curr_tr, Error):
            logger.error(f"Failed to save transformation: {curr_tr.description}")
            return
        name = self._ctx.state.text_input_transformation_name
        err = self._ctx.transformation_save_manager.set(  #
            source_uid=ct_uid,  #
            destination_uid=xray_uid,  #
            name=name,  #
            transformation=curr_tr,  #
        )
        if isinstance(err, Error):
            logger.error(f"Error saving transformation to idx ({ct_uid}, {xray_uid}, {name})' to save "
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
        res: tuple[str, str] | Error = self._get_ct_xray_uids()
        if isinstance(res, Error):
            logger.error(f"Failed to load transformation: {res.description}")
            return
        ct_uid, xray_uid = res
        tr: Transformation | Error = self._ctx.transformation_save_manager.get_transformation(  #
            source_uid=ct_uid,  #
            destination_uid=xray_uid,  #
            name=name,  #
        )
        if isinstance(tr, Error):
            logger.error(f"Error loading transformation of idx '({ct_uid}, {xray_uid}, {name})' from save manager: "
                         f"{tr.description}")
            return
        curr_tr: Transformation | Error = self._ctx.dadg.get(self._c_t_key)
        if isinstance(curr_tr, Error):
            logger.error(f"No current transformation whose device to copy: {curr_tr.description}")
            return
        device = curr_tr.rotation.device
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
        res: tuple[str, str] | Error = self._get_ct_xray_uids()
        if isinstance(res, Error):
            logger.error(f"Failed to delete transformation: {res.description}")
            return
        ct_uid, xray_uid = res
        err = self._ctx.transformation_save_manager.remove(  #
            source_uid=ct_uid,  #
            destination_uid=xray_uid,  #
            name=name,  #
        )
        if isinstance(err, Error):
            logger.error(f"Error deleting transformation of idx '({ct_uid}, {xray_uid}, {name})' from save manager: "
                         f"{err.description}")
        self._update_saved_transformation_names()
