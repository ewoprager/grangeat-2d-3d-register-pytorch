from traitlets import HasTraits, Int, Float, Instance, Bool, Enum, Unicode, Undefined, observe
from typing import Any

from reg23_experiments.utils.data import StrictHasTraits
from reg23_experiments.ops.data_manager import data_manager


class NoParameters(StrictHasTraits):
    pass


class PsoParameters(StrictHasTraits):
    particle_count: int = Int(min=1, default_value=2000).tag(ui=True)
    starting_spread: float = Float(min=0.0, default_value=1.0).tag(ui=True)
    inertia_coefficient: float = Float(min=0.0, default_value=0.28).tag(ui=True)
    cognitive_coefficient: float = Float(min=0.0, default_value=2.525).tag(ui=True)
    social_coefficient: float = Float(min=0.0, default_value=1.225).tag(ui=True)


class LocalSearchParameters(StrictHasTraits):
    initial_step_size: float = Float(min=0.0, default_value=0.1).tag(ui=True)
    no_improvement_threshold: int = Int(min=0, default_value=10).tag(ui=True)
    step_size_reduction_ratio: float = Float(min=0.0, max=1.0, default_value=0.75).tag(ui=True)
    max_reductions: int = Int(min=0, default_value=4).tag(ui=True)
    max_iterations: int = Int(min=1, default_value=5000).tag(ui=True)


class LocalZnccParameters(StrictHasTraits):
    kernel_size: int = Int(min=1, default_value=8).tag(ui=True)


class Parameters(StrictHasTraits):
    ct_path: str = Unicode(default_value=Undefined).tag(ui=True)
    downsample_level: int = Int(min=0, default_value=Undefined).tag(ui=True)
    truncation_percent: int = Int(min=0, max=100, default_value=Undefined).tag(ui=True)
    cropping: str = Enum(values=[  #
        "None",  #
        "nonzero_drr",  #
        "full_depth_drr"  #
    ]).tag(ui=True)
    mask: str = Enum(values=[  #
        "None",  #
        "Every evaluation",  #
        "Every evaluation weighting zncc"  #
    ], default_value=Undefined).tag(ui=True)
    sim_metric: str = Enum(values=[  #
        "zncc",  #
        "local_zncc",  #
        "multiscale_zncc",  #
        "gradient_correlation"  #
    ], default_value=Undefined).tag(ui=True)
    sim_metric_parameters: HasTraits = Instance(HasTraits, allow_none=False, default_value=Undefined).tag(ui=True)
    starting_distance: float = Float(min=0.0)
    sample_count_per_distance: int = Int(min=1)
    optimisation_algorithm: str = Enum(values=[  #
        "pso",  #
        "local_search"  #
    ], default=Undefined).tag(ui=True)
    op_algo_parameters: HasTraits = Instance(HasTraits, allow_none=False, default_value=Undefined).tag(ui=True)
    iteration_count: int = Int(min=0).tag(ui=True)

    def __init__(self, **kwargs):
        self._op_algo_cache: dict[str, Any] = {}
        self._sim_metric_cache: dict[str, Any] = {}
        super().__init__(**kwargs)
        self._update_op_algo_params()
        self._update_sim_metric_params()

    @observe("optimisation_algorithm")
    def _op_algo_changed(self, _):
        self._update_op_algo_params()

    @observe("sim_metric")
    def _sim_metric_changed(self, _):
        self._update_sim_metric_params()

    def _update_op_algo_params(self):
        if self.optimisation_algorithm not in self._op_algo_cache:
            if self.optimisation_algorithm == "pso":
                self._op_algo_cache[self.optimisation_algorithm] = PsoParameters()
            elif self.optimisation_algorithm == "local_search":
                self._op_algo_cache[self.optimisation_algorithm] = LocalSearchParameters()
            else:
                raise ValueError(f"Unrecognised optimisation algorithm '{self.optimisation_algorithm}'.")

        self.op_algo_parameters = self._op_algo_cache[self.optimisation_algorithm]

    def _update_sim_metric_params(self):
        if self.sim_metric not in self._sim_metric_cache:
            if self.sim_metric == "zncc":
                self._sim_metric_cache[self.sim_metric] = NoParameters()
            elif self.sim_metric == "local_zncc":
                self._sim_metric_cache[self.sim_metric] = LocalZnccParameters()
            elif self.sim_metric == "multiscale_zncc":
                self._sim_metric_cache[self.sim_metric] = NoParameters()
            elif self.sim_metric == "gradient_correlation":
                self._sim_metric_cache[self.sim_metric] = NoParameters()
            else:
                raise ValueError(f"Unrecognised optimisation algorithm '{self.sim_metric}'.")

        self.sim_metric_parameters = self._sim_metric_cache[self.sim_metric]
