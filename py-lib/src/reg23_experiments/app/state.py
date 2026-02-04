from traitlets import HasTraits, Instance, Bool, Unicode

from reg23_experiments.ops.data_manager import DAG
from reg23_experiments.experiments.parameters import Parameters

from ._gui_param_to_dag_node import respond_to_crop_change, respond_to_mask_change

__all__ = ["AppState"]


class AppState(HasTraits):
    parameters: Parameters = Instance(Parameters, allow_none=False)

    dag: DAG = Instance(DAG, allow_none=False)

    button_evaluate_once: bool = Bool(default_value=False)
    eval_once_result: str | None = Unicode(allow_none=True, default_value=None)
    job_state_description: str | None = Unicode(allow_none=True, default_value=None)
    button_run_one_iteration: bool = Bool(default_value=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.parameters.observe(self._update_dag_ct_path, names=["ct_path"])
        self.dag.set_data("ct_path", self.parameters.ct_path)
        self.parameters.observe(self._update_dag_downsample_level, names=["downsample_level"])
        self.dag.set_data("downsample_level", self.parameters.downsample_level)
        self.parameters.observe(self._update_dag_truncation_percent, names=["truncation_percent"])
        self.dag.set_data("truncation_percent", self.parameters.truncation_percent)
        self.parameters.observe(lambda change: respond_to_mask_change(self.dag, change), names=["mask"])
        self.parameters.observe(lambda change: respond_to_crop_change(self.dag, change), names=["cropping"])

    def _update_dag_ct_path(self, change) -> None:
        self.dag.set_data("ct_path", change.new, check_equality=True)

    def _update_dag_downsample_level(self, change) -> None:
        self.dag.set_data("downsample_level", change.new, check_equality=True)

    def _update_dag_truncation_percent(self, change) -> None:
        self.dag.set_data("truncation_percent", change.new, check_equality=True)
