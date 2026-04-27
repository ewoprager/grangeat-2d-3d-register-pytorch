from typing import Literal

import torch
from timm.loss import asymmetric_loss
from traitlets import Bool, Enum, HasTraits, Instance, Int, List, Unicode, Union

from reg23_experiments.app.gui_settings import GUISettings
from reg23_experiments.experiments.parameters import Parameters
from reg23_experiments.experiments.drr_params import DRRParams

__all__ = ["AppState", "WorkerState"]


class WorkerState(HasTraits):
    """
    Only contains data; either simple values, or other `HasTraits` instances that themselves just contain data.
    """
    current_best_f: torch.Tensor = Instance(torch.Tensor, allow_none=True)
    current_best_x: torch.Tensor = Instance(torch.Tensor, allow_none=True)
    iteration: int | Literal["initialising", "finished"] | None = Union(
        trait_types=[Int(), Enum(values=["initialising", "finished"])], allow_none=True, default_value=None)
    max_iterations: int = Int()


class AppState(HasTraits):
    """
    Only contains data; either simple values, or other `HasTraits` instances that themselves just contain data.
    """
    gui_settings: GUISettings = Instance(GUISettings, allow_none=False, default_value=GUISettings())

    parameters: Parameters = Instance(Parameters, allow_none=False)

    button_open_ct_file: bool = Bool(default_value=False)
    button_open_ct_dir: bool = Bool(default_value=False)

    button_open_xray_file: bool = Bool(default_value=False)
    button_unload_xray_file: bool = Bool(default_value=False)
    unload_xray_choice: str | None = Unicode(allow_none=True, default_value=None)

    button_evaluate_once: bool = Bool(default_value=False)
    eval_once_result: str | None = Unicode(allow_none=True, default_value=None)

    worker_state: WorkerState | None = Instance(WorkerState, allow_none=True, default_value=None)
    button_run_one_iteration: bool = Bool(default_value=False)
    button_run: bool = Bool(default_value=False)
    button_load_current_best: bool = Bool(default_value=False)

    register_xray_choice: str | None = Unicode(allow_none=True, default_value=None)
    saved_transformation_names: list[str] = List(trait=Unicode(), default_value=[])
    text_input_transformation_name: str = Unicode(default_value="")
    button_save_transformation: bool = Bool(default_value=False)
    button_load_transformation_of_name: str | None = Unicode(allow_none=True, default_value=None)
    button_delete_transformation_of_name: str | None = Unicode(allow_none=True, default_value=None)

    drr_params: DRRParams = Instance(DRRParams, allow_none=False, default_value=DRRParams())
    drr_name_input: str = Unicode(allow_none=False, default_value="drr")
    button_create_drr: bool = Bool(default_value=False)
