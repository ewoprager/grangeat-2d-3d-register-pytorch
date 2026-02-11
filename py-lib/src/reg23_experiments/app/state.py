from traitlets import HasTraits, Instance, Float, Int, Bool, Unicode, validate, Enum, TraitError, List, Union, All
from typing import Literal

import pathlib
import torch

from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph
from reg23_experiments.experiments.parameters import Parameters

from ._gui_param_to_dag_node import respond_to_crop_change, respond_to_mask_change, respond_to_crop_value_change, \
    respond_to_crop_value_value_change

__all__ = ["AppState", "WorkerState"]


class WorkerState(HasTraits):
    current_best_f: torch.Tensor = Instance(torch.Tensor, allow_none=True)
    current_best_x: torch.Tensor = Instance(torch.Tensor, allow_none=True)
    iteration: int | Literal["initialising", "finished"] | None = Union(
        trait_types=[Int(), Enum(values=["initialising", "finished"])], allow_none=True, default_value=None)
    max_iterations: int = Int()


class AppState(HasTraits):
    parameters: Parameters = Instance(Parameters, allow_none=False)

    dadg: DirectedAcyclicDataGraph = Instance(DirectedAcyclicDataGraph, allow_none=False)

    button_evaluate_once: bool = Bool(default_value=False)
    eval_once_result: str | None = Unicode(allow_none=True, default_value=None)

    worker_state: WorkerState | None = Instance(WorkerState, allow_none=True, default_value=None)
    button_run_one_iteration: bool = Bool(default_value=False)
    button_run: bool = Bool(default_value=False)
    button_load_current_best: bool = Bool(default_value=False)

    transformation_save_directory: pathlib.Path = Instance(pathlib.Path, allow_none=False)
    saved_transformation_names: list[str] = List(trait=Unicode(), default_value=[])
    text_input_transformation_name: str = Unicode(default_value="")
    button_save_transformation: bool = Bool(default_value=False)
    button_load_transformation_of_name: str | None = Unicode(allow_none=True, default_value=None)
    button_delete_transformation_of_name: str | None = Unicode(allow_none=True, default_value=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.parameters.observe(self._update_dag_ct_path, names=["ct_path"])
        self.dadg.set("ct_path", self.parameters.ct_path)
        self.parameters.observe(self._update_dag_downsample_level, names=["downsample_level"])
        self.dadg.set("downsample_level", self.parameters.downsample_level)
        self.parameters.observe(self._update_dag_truncation_percent, names=["truncation_percent"])
        self.dadg.set("truncation_percent", self.parameters.truncation_percent)
        self.parameters.observe(self._update_dag_target_flipped, names=["target_flipped"])
        self.dadg.set("target_flipped", self.parameters.target_flipped)
        self.parameters.observe(lambda change: respond_to_mask_change(self.dadg, change), names=["mask"])
        self.parameters.observe(lambda change: respond_to_crop_change(self.dadg, change), names=["cropping"])
        self.parameters.observe(lambda change: respond_to_crop_value_change(self.dadg, change),
                                names=["cropping_value"])

    def _update_dag_ct_path(self, change) -> None:
        self.dadg.set("ct_path", change.new, check_equality=True)

    def _update_dag_downsample_level(self, change) -> None:
        self.dadg.set("downsample_level", change.new, check_equality=True)

    def _update_dag_truncation_percent(self, change) -> None:
        self.dadg.set("truncation_percent", change.new, check_equality=True)

    def _update_dag_target_flipped(self, change) -> None:
        self.dadg.set("target_flipped", change.new, check_equality=True)

    @validate("transformation_save_directory")
    def _validate_transformation_save_directory(self, proposal):
        if not proposal["value"].is_dir():
            raise TraitError("'transformation_save_directory' must be a valid directory")
        return proposal["value"]
