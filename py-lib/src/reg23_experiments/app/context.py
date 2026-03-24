import logging
from traitlets import TraitError

import pathlib

from reg23_experiments.app.state import AppState
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph, NoNodeData
from reg23_experiments.data.electrode_save_data import ElectrodeSaveManager
from reg23_experiments.experiments.parameters import Parameters
from reg23_experiments.data.transformation_save_data import TransformationSaveManager
from reg23_experiments.app.cache_manager import CacheManager
from reg23_experiments.data.structs import Error
from reg23_experiments.app.gui.viewer_singleton import viewer

from ._gui_param_to_dag_node import respond_to_crop_change, respond_to_mask_change, respond_to_crop_value_change, \
    respond_to_crop_value_value_change

__all__ = ["AppContext"]

logger = logging.getLogger(__name__)


class AppContext:
    def __init__(self, *, parameters: Parameters, dadg: DirectedAcyclicDataGraph,
                 electrode_save_directory: pathlib.Path, transformation_save_directory: pathlib.Path):
        self._state = AppState(parameters=parameters)
        self._dadg = dadg
        self._cache_manager = CacheManager()
        self._electrode_save_manager = ElectrodeSaveManager(electrode_save_directory)
        self._transformation_save_manager = TransformationSaveManager(transformation_save_directory)

        # load values from the cache
        if self.dadg.has_node("ct_path"):
            res = self.dadg.get("ct_path")
            if not isinstance(res, Error):
                self._state.ct_path = self.dadg.get("ct_path")
        if not self._state.ct_path:
            last_ct_path: pathlib.Path | None = self._cache_manager.last_ct_path
            if last_ct_path is not None:
                self._state.ct_path = str(last_ct_path)
                self.dadg.set("ct_path", self._state.ct_path, check_equality=True)

        # set up observers such that parameters changed in the UI effect the DADG and cache correctly
        self._state.observe(self._ct_path_changed, names=["ct_path"])
        self._dadg.set("ct_path", NoNodeData if self._state.ct_path is None else self._state.ct_path)
        self._state.parameters.observe(self._update_dag_downsample_level, names=["downsample_level"])
        self._dadg.set("downsample_level", self._state.parameters.downsample_level)
        self._state.parameters.observe(self._update_dag_truncation_percent, names=["truncation_percent"])
        self._dadg.set("truncation_percent", self._state.parameters.truncation_percent)
        self._state.parameters.observe(self._update_dag_target_flipped, names=["target_flipped"])
        self._dadg.set("a__target_flipped", self._state.parameters.target_flipped)
        self._state.parameters.observe(lambda change: respond_to_mask_change(self.dadg, change), names=["mask"])
        self._state.parameters.observe(lambda change: respond_to_crop_change(self.dadg, change), names=["cropping"])
        self._state.parameters.observe(lambda change: respond_to_crop_value_change(self.dadg, change),
                                       names=["cropping_value"])
        self._state.observe(self._button_open_ct_file, names=["button_open_ct_file"])
        self._state.observe(self._button_open_ct_dir, names=["button_open_ct_dir"])

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def dadg(self) -> DirectedAcyclicDataGraph:
        return self._dadg

    @property
    def electrode_save_manager(self) -> ElectrodeSaveManager:
        return self._electrode_save_manager

    @property
    def transformation_save_manager(self) -> TransformationSaveManager:
        return self._transformation_save_manager

    def _ct_path_changed(self, change) -> None:
        self.dadg.set("ct_path", NoNodeData if change.new is None else change.new, check_equality=True)
        self._cache_manager.last_ct_path = change.new

    def _update_dag_downsample_level(self, change) -> None:
        self.dadg.set("downsample_level", change.new, check_equality=True)

    def _update_dag_truncation_percent(self, change) -> None:
        self.dadg.set("truncation_percent", change.new, check_equality=True)

    def _update_dag_target_flipped(self, change) -> None:
        self.dadg.set("a__target_flipped", change.new, check_equality=True)

    def _button_open_ct_file(self, change) -> None:
        if not change.new:
            return
        self.state.button_open_ct = False

        from qtpy.QtWidgets import QFileDialog
        file, _ = QFileDialog.getOpenFileName(viewer().window._qt_window, "Open a CT volume file")
        logger.info(f"Opening CT volume file '{file}'")
        self._open_ct_path(file)

    def _button_open_ct_dir(self, change) -> None:
        if not change.new:
            return
        self.state.button_open_ct = False

        from qtpy.QtWidgets import QFileDialog
        dire = QFileDialog.getExistingDirectory(viewer().window._qt_window,
                                                "Open a directory of DICOM files as slices of a CT volume")
        logger.info(f"Opening DICOM files in directory '{dire}' as slices of CT volume")
        self._open_ct_path(dire)

    def _open_ct_path(self, path: str) -> None:
        try:
            self._state.ct_path = path
        except TraitError:
            logger.warning(f"CT path not valid: '{path}'")
            return
