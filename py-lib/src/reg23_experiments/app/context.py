import pathlib

from reg23_experiments.app.state import AppState
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph
from reg23_experiments.data.electrode_save_data import ElectrodeSaveManager
from reg23_experiments.experiments.parameters import Parameters
from reg23_experiments.data.transformation_save_data import TransformationSaveManager

from ._gui_param_to_dag_node import respond_to_crop_change, respond_to_mask_change, respond_to_crop_value_change, \
    respond_to_crop_value_value_change

__all__ = ["AppContext"]


class AppContext:
    def __init__(self, *, parameters: Parameters, dadg: DirectedAcyclicDataGraph,
                 electrode_save_directory: pathlib.Path, transformation_save_directory: pathlib.Path):
        self._state = AppState(parameters=parameters)
        self._dadg = dadg
        self._electrode_save_manager = ElectrodeSaveManager(electrode_save_directory)
        self._transformation_save_manager = TransformationSaveManager(transformation_save_directory)

        self._state.parameters.observe(self._update_dag_ct_path, names=["ct_path"])
        self._dadg.set("ct_path", self._state.parameters.ct_path)
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

    def _update_dag_ct_path(self, change) -> None:
        self.dadg.set("ct_path", change.new, check_equality=True)

    def _update_dag_downsample_level(self, change) -> None:
        self.dadg.set("downsample_level", change.new, check_equality=True)

    def _update_dag_truncation_percent(self, change) -> None:
        self.dadg.set("truncation_percent", change.new, check_equality=True)

    def _update_dag_target_flipped(self, change) -> None:
        self.dadg.set("a__target_flipped", change.new, check_equality=True)
