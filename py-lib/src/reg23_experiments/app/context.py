import logging
from traitlets import TraitError
from typing import Any

import pathlib
from magicgui.widgets import request_values
import torch

from reg23_experiments.app.state import AppState
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph, NoNodeData, updaters, capture_in_namespaces
from reg23_experiments.data.electrode_save_data import ElectrodeSaveManager
from reg23_experiments.experiments.parameters import Parameters
from reg23_experiments.data.transformation_save_data import TransformationSaveManager
from reg23_experiments.app.cache_manager import CacheManager
from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.app.gui.viewer_singleton import viewer
from reg23_experiments.experiments.parameters import XrayParameters
from reg23_experiments.app.gui.input_manager import InputManager
from reg23_experiments.io.serialize import deserialize_recursive, serialize_recursive
from reg23_experiments.utils.data import observe_all_traits_recursively

from reg23_experiments.experiments.multi_xray_truncation_updaters import set_target_image, project_drr, read_xray_uid

from ._gui_param_to_dag_node import respond_to_crop_change, respond_to_mask_change, respond_to_crop_value_change, \
    respond_to_crop_value_value_change

__all__ = ["AppContext"]

logger = logging.getLogger(__name__)


class AppContext:
    def __init__(self, *, parameters: Parameters, dadg: DirectedAcyclicDataGraph,
                 electrode_save_directory: pathlib.Path, transformation_save_directory: pathlib.Path):
        self._input_manager = InputManager()
        self._state = AppState(parameters=parameters)
        self._dadg = dadg
        self._cache_manager = CacheManager()
        self._electrode_save_manager = ElectrodeSaveManager(electrode_save_directory)
        self._transformation_save_manager = TransformationSaveManager(transformation_save_directory)

        # load params from the cache
        res: dict[str, Any] | None = self._cache_manager.last_params
        if res is not None:
            self.state.parameters = deserialize_recursive(value=res, old_value=self.state.parameters)

        # load X-ray files according to the parameters
        for name, value in self._state.parameters.xray_parameters.items():
            self._add_xray(name=name, file_path=value.file_path)

        # set up observers such that parameters changed in the UI effect the DADG and cache correctly
        self._state.parameters.observe(self._ct_path_changed, names=["ct_path"])
        # load CT files according to the parameters
        self._dadg.set("ct_path",
                       NoNodeData if self._state.parameters.ct_path is None else self._state.parameters.ct_path)
        self._state.parameters.observe(self._update_dag_downsample_level, names=["downsample_level"])
        self._dadg.set("downsample_level", self._state.parameters.downsample_level)
        self._state.parameters.observe(self._update_dag_truncation_percent, names=["truncation_percent"])
        self._dadg.set("truncation_percent", self._state.parameters.truncation_percent)
        # self._state.parameters.observe(self._update_dag_target_flipped, names=["target_flipped"])
        # self._dadg.set("a__target_flipped", self._state.parameters.target_flipped)
        self._state.parameters.observe(lambda change: respond_to_mask_change(self.dadg, change), names=["mask"])
        # self._state.parameters.observe(lambda change: respond_to_crop_change(self.dadg, change), names=["cropping"])
        # self._state.parameters.observe(lambda change: respond_to_crop_value_change(self.dadg, change),
        #                                names=["cropping_value"])
        self._state.observe(self._button_open_ct_file, names=["button_open_ct_file"])
        self._state.observe(self._button_open_ct_dir, names=["button_open_ct_dir"])
        self._state.observe(self._button_open_xray_file, names=["button_open_xray_file"])
        observe_all_traits_recursively(self._any_parameter_changed, self._state.parameters)

    @property
    def input_manager(self) -> InputManager:
        return self._input_manager

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

    def _update_dag_downsample_level(self, change) -> None:
        self.dadg.set("downsample_level", change.new, check_equality=True)

    def _update_dag_truncation_percent(self, change) -> None:
        self.dadg.set("truncation_percent", change.new, check_equality=True)

    # def _update_dag_target_flipped(self, change) -> None:
    #     self.dadg.set("a__target_flipped", change.new, check_equality=True)

    def _button_open_ct_file(self, change) -> None:
        if not change.new:
            return
        self.state.button_open_ct = False

        from qtpy.QtWidgets import QFileDialog
        file, _ = QFileDialog.getOpenFileName(viewer().window._qt_window, "Open a CT volume file")
        if not file:
            return
        logger.info(f"Opening CT volume file '{file}'")
        self._open_ct_path(file)

    def _button_open_ct_dir(self, change) -> None:
        if not change.new:
            return
        self.state.button_open_ct = False

        from qtpy.QtWidgets import QFileDialog
        dire = QFileDialog.getExistingDirectory(viewer().window._qt_window,
                                                "Open a directory of DICOM files as slices of a CT volume")
        if not dire:
            return
        logger.info(f"Opening DICOM files in directory '{dire}' as slices of CT volume")
        self._open_ct_path(dire)

    def _open_ct_path(self, path: str) -> None:
        try:
            self._state.parameters.ct_path = path
        except TraitError:
            logger.warning(f"CT path not valid: '{path}'")
            return

    def _button_open_xray_file(self, change) -> None:
        if not change.new:
            return
        self.state.button_open_xray_file = False

        # Get the user to choose a file
        from qtpy.QtWidgets import QFileDialog
        file, _ = QFileDialog.getOpenFileName(viewer().window._qt_window, "Open a CT volume file")
        if not file:
            return
        logger.info(f"Opening X-ray image file '{file}'")
        for _, ps in self._state.parameters.xray_parameters.items():
            if ps.file_path == file:
                logger.warning(f"X-ray '{file}' is already open.")
                return

        # Get a valid, unique name for the X-ray from the user
        prompt_string = "Enter a unique name for the X-ray"
        name = pathlib.Path(file).stem[:6]
        if name not in self._state.parameters.xray_parameters:
            prompt_string += f"; leave empty for default value '{name}' generated from file path"
        prompt_string += ":"
        values = request_values(name={"annotation": str, "label": prompt_string})
        if values["name"]:
            name = values["name"]
        while not name or name in self._state.parameters.xray_parameters:
            values = request_values(name={"annotation": str, "label": prompt_string})
            if values["name"]:
                name = values["name"]
        self._add_xray(name=name, file_path=file)

    def _add_xray(self, *, name: str, file_path: str):
        # Get the X-ray parameters
        p = XrayParameters(file_path=file_path)
        self.state.parameters.xray_parameters = {**self.state.parameters.xray_parameters, name: p}
        self.dadg.set(f"{name}__xray_path", file_path)
        self.dadg.set(f"{name}__target_flipped", p.target_flipped)
        self.dadg.set(f"{name}__source_offset", torch.zeros(2))
        self.dadg.set(f"{name}__mask_transformation", None)
        self.dadg.set(f"{name}__current_transformation", Transformation.zero(device=self.dadg.get("device")))
        namespace_captures = {key: name for key in ["image_2d_full", "fixed_image_spacing", "transformation_gt",  #
                                                    "source_distance", "xray_path", "target_flipped", "moving_image",
                                                    "fixed_image_size", "fixed_image_offset", "xray_sop_instance_uid",
                                                    "fixed_image", "cropped_target", "mask", "translation_offset",
                                                    "image_2d_scale_factor", "source_offset", "mask_transformation",
                                                    "current_transformation"]}
        # add namespaced updaters
        err = self.dadg.add_updater(f"{name}__refresh_image_2d_scale_factor",
                                    capture_in_namespaces(namespace_captures)(updaters.refresh_image_2d_scale_factor))
        if isinstance(err, Error):
            logger.error(f"Error adding updater: {err.description}")
        err = self.dadg.add_updater(f"{name}__refresh_hyperparameter_dependent",
                                    capture_in_namespaces(namespace_captures)(
                                        updaters.refresh_hyperparameter_dependent))
        if isinstance(err, Error):
            logger.error(f"Error adding updater: {err.description}")
        err = self.dadg.add_updater(f"{name}__refresh_mask_transformation_dependent",
                                    capture_in_namespaces(namespace_captures)(
                                        updaters.refresh_mask_transformation_dependent))
        if isinstance(err, Error):
            logger.error(f"Error adding updater: {err.description}")
        err = self.dadg.add_updater(f"{name}__project_drr", capture_in_namespaces(namespace_captures)(project_drr))
        if isinstance(err, Error):
            logger.error(f"Error adding updater: {err.description}")
        err = self.dadg.add_updater(f"{name}__xray_uid", capture_in_namespaces(namespace_captures)(read_xray_uid))
        if isinstance(err, Error):
            logger.error(f"Error adding updater: {err.description}")
        err = self.dadg.add_updater(f"{name}__set_target_image",
                                    capture_in_namespaces(namespace_captures)(set_target_image))
        if isinstance(err, Error):
            logger.error(f"Error adding updater: {err.description}")

        err = self.dadg.get(f"{name}__moving_image")
        if isinstance(err, Error):
            logger.error(f"Failed to get moving image '{name}': {err.description}")

    def _any_parameter_changed(self, change) -> None:
        self._cache_manager.last_params = serialize_recursive(self.state.parameters)
