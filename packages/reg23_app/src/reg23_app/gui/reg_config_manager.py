import logging

from reg23_app.state import AppState
from reg23_experiments.data.structs import Cropping, Error
from reg23_experiments.data.xray_reg_save_data import XRayRegSaveManager
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph

__all__ = ["RegConfigManager"]

logger = logging.getLogger(__name__)


class RegConfigManager:
    """
    Reads from and write to the state and dadg only
    """

    def __init__(self, state: AppState, dadg: DirectedAcyclicDataGraph, xray_reg_save_manager: XRayRegSaveManager):
        self._state = state
        self._dadg = dadg
        self._xray_reg_save_manager = xray_reg_save_manager

        self._state.observe(self._button_save_xray_reg_config, names=["button_save_xray_reg_config"])

    def _button_save_xray_reg_config(self, change) -> None:
        if not change.new:
            return
        self._state.button_save_xray_reg_config = False

        namespace: str | None = self._state.save_xray_reg_config_choice
        if namespace is None:
            logger.warning(f"Couldn't save reg config as no X-ray is selected.")
            return

        # Save the X-ray reg configs
        uid: str | Error = self._dadg.get(f"{namespace}__xray_sop_instance_uid")
        if isinstance(uid, Error):
            logger.error(f"Couldn't save reg config for X-ray '{namespace}'; couldn't get UID: {uid.description}")
            return
        cropping: Cropping | None | Error = self._dadg.get(f"{namespace}__cropping")
        if isinstance(cropping, Error):
            logger.error(
                f"Couldn't save reg config for X-ray '{namespace}'; couldn't get cropping: {cropping.description}")
            return
        target_flipped: bool | Error = self._dadg.get(f"{namespace}__target_flipped")
        if isinstance(target_flipped, Error):
            logger.error(f"Couldn't save reg config for X-ray '{namespace}'; couldn't get target_flipped: "
                         f"{target_flipped.description}")
            return
        if isinstance(err := self._xray_reg_save_manager.set(  #
                uid=str(uid),  #
                flipped=target_flipped,  #
                cropping=Cropping() if cropping is None else cropping,  #
        ), Error):
            logger.error(f"Error saving reg config for X-ray '{uid}': {err.description}")
        logger.info(f"Cropping saved for X-ray '{uid}'.")
