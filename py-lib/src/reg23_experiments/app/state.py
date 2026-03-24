from traitlets import HasTraits, Instance, Float, Int, Bool, Unicode, Enum, List, Union, validate, TraitError
from typing import Literal

import torch
import pathlib

from reg23_experiments.experiments.parameters import Parameters
from reg23_experiments.app.gui_settings import GUISettings

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

    ct_path: str | None = Unicode(allow_none=True, default_value=None).tag(ui=True)

    button_evaluate_once: bool = Bool(default_value=False)
    eval_once_result: str | None = Unicode(allow_none=True, default_value=None)

    worker_state: WorkerState | None = Instance(WorkerState, allow_none=True, default_value=None)
    button_run_one_iteration: bool = Bool(default_value=False)
    button_run: bool = Bool(default_value=False)
    button_load_current_best: bool = Bool(default_value=False)

    saved_transformation_names: list[str] = List(trait=Unicode(), default_value=[])
    text_input_transformation_name: str = Unicode(default_value="")
    button_save_transformation: bool = Bool(default_value=False)
    button_load_transformation_of_name: str | None = Unicode(allow_none=True, default_value=None)
    button_delete_transformation_of_name: str | None = Unicode(allow_none=True, default_value=None)

    @validate("ct_path")
    def _validate_ct_path(self, proposal):
        if proposal["value"] is None:
            return proposal["value"]
        if not pathlib.Path(proposal["value"]).is_dir():
            raise TraitError("Invalid CT path")
        return proposal["value"]
