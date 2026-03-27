from traitlets import HasTraits, Int, Float, Instance, Bool, Enum, Unicode, Undefined, observe, Dict, validate, \
    TraitError, Union
from typing import Any, Literal

import pathlib

from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph
from reg23_experiments.data.structs import Cropping, Error

__all__ = ["PsoParameters", "LocalZnccParameters", "LocalSearchParameters", "Parameters", "Context", "XrayParameters"]


class XrayParameters(HasTraits):
    """
    Only contains data; either simple values, or other `HasTraits` instances that themselves just contain data.
    """
    file_path: str = Unicode(allow_none=False).tag(ui=True)
    target_flipped: bool = Bool(allow_none=False, default_value=False).tag(ui=True)
    cropping: Literal["None", "Fixed"] = Enum(values=[  #
        "None",  #
        "Fixed"  #
    ], allow_none=False, default_value="None").tag(ui=True)
    cropping_value: Cropping | None = Instance(Cropping, allow_none=True, default_value=None).tag(ui=True)

    def __init__(self, **kwargs):
        self._cropping_cache: Cropping | None = None
        super().__init__(**kwargs)
        self._update_cropping_value()

    @observe("cropping")
    def _cropping_changed(self, _):
        self._update_cropping_value()

    @validate("file_path")
    def _validate_file_path(self, proposal):
        if not pathlib.Path(proposal["value"]).is_file():
            raise TraitError("Invalid X-ray path")
        return proposal["value"]

    def _update_cropping_value(self):
        if self.cropping == "Fixed":
            if isinstance(self.cropping_value, Cropping):
                return
            if self._cropping_cache is None:
                self._cropping_cache = Cropping()
            self.cropping_value = self._cropping_cache
        else:  # cropping is "None"
            if self.cropping_value is None:
                return
            self._cropping_cache = self.cropping_value
            self.cropping_value = None


class PsoParameters(HasTraits):
    """
    Only contains data; either simple values, or other `HasTraits` instances that themselves just contain data.
    """
    particle_count: int = Int(min=1, default_value=2000).tag(ui=True)
    starting_spread: float = Float(min=0.0, default_value=1.0).tag(ui=True)
    inertia_coefficient: float = Float(min=0.0, default_value=0.28).tag(ui=True)
    cognitive_coefficient: float = Float(min=0.0, default_value=2.525).tag(ui=True)
    social_coefficient: float = Float(min=0.0, default_value=1.225).tag(ui=True)


class LocalSearchParameters(HasTraits):
    """
    Only contains data; either simple values, or other `HasTraits` instances that themselves just contain data.
    """
    initial_step_size: float = Float(min=0.0, default_value=0.1).tag(ui=True)
    no_improvement_threshold: int = Int(min=0, default_value=10).tag(ui=True)
    step_size_reduction_ratio: float = Float(min=0.0, max=1.0, default_value=0.75).tag(ui=True)
    max_reductions: int = Int(min=0, default_value=4).tag(ui=True)
    max_iterations: int = Int(min=1, default_value=5000).tag(ui=True)


class CmaesParameters(HasTraits):
    """
    Only contains data; either simple values, or other `HasTraits` instances that themselves just contain data.
    """
    pass


class LocalZnccParameters(HasTraits):
    """
    Only contains data; either simple values, or other `HasTraits` instances that themselves just contain data.
    """
    kernel_size: int = Int(min=1, default_value=8).tag(ui=True)


class Parameters(HasTraits):
    """
    Only contains data; either simple values, or other `HasTraits` instances that themselves just contain data.
    """
    ct_path: str | None = Unicode(allow_none=True, default_value=None).tag(ui=True)
    downsample_level: int = Int(min=0).tag(ui=True)
    truncation_percent: int = Int(min=0, max=100).tag(ui=True)
    mask: Literal["None", "Every evaluation", "Every evaluation weighting zncc"] = Enum(values=[  #
        "None",  #
        "Every evaluation",  #
        "Every evaluation weighting zncc"  #
    ]).tag(ui=True)
    sim_metric: Literal["zncc", "local_zncc", "multiscale_zncc", "gradient_correlation"] = Enum(values=[  #
        "zncc",  #
        "local_zncc",  #
        "multiscale_zncc",  #
        "gradient_correlation"  #
    ]).tag(ui=True)
    sim_metric_parameters: LocalZnccParameters | None = Instance(LocalZnccParameters, allow_none=True,
                                                                 default_value=None).tag(ui=True)
    starting_distance: float = Float(min=0.0)
    sample_count_per_distance: int = Int(min=1)
    optimisation_algorithm: Literal["pso", "local_search", "cmaes"] = Enum(values=[  #
        "pso",  #
        "local_search",  #
        "cmaes",  #
    ], default=Undefined).tag(ui=True)
    op_algo_parameters: PsoParameters | LocalSearchParameters = Union(
        trait_types=[Instance(PsoParameters, allow_none=False), Instance(PsoParameters, allow_none=False)]).tag(ui=True)
    iteration_count: int = Int(min=0).tag(ui=True)
    xray_parameters: dict[str, XrayParameters] = Dict(  #
        key_trait=Unicode(allow_none=False),  #
        value_trait=Instance(XrayParameters, allow_none=False),  #
        allow_none=False,  #
        default_value=dict({})  #
    ).tag(ui=True)

    OP_ALGO_PARAM_CLASSES: dict[str, type] = {  #
        "pso": PsoParameters,  #
        "local_search": LocalSearchParameters,  #
    }

    SIM_MET_PARAM_CLASSES: dict[str, type] = {  #
        "zncc": type(None),  #
        "local_zncc": LocalZnccParameters,  #
        "multiscale_zncc": type(None),  #
        "gradient_correlation": type(None),  #
    }

    def __init__(self, **kwargs):
        self._op_algo_cache: dict[str, Any] = {}
        self._sim_metric_cache: dict[str, Any] = {}
        super().__init__(**kwargs)
        self._update_op_algo_params()
        self._update_sim_metric_params()

    @validate("ct_path")
    def _validate_ct_path(self, proposal):
        if proposal["value"] is None:
            return proposal["value"]
        if not pathlib.Path(proposal["value"]).exists():
            raise TraitError("Invalid CT path")
        return proposal["value"]

    @observe("optimisation_algorithm")
    def _op_algo_changed(self, _):
        self._update_op_algo_params()

    @observe("sim_metric")
    def _sim_metric_changed(self, _):
        self._update_sim_metric_params()

    def _update_op_algo_params(self):
        desired_cls: type = Parameters.OP_ALGO_PARAM_CLASSES[self.optimisation_algorithm]
        if isinstance(self.op_algo_parameters, desired_cls):
            self._op_algo_cache[self.optimisation_algorithm] = self.op_algo_parameters
        else:
            if self.optimisation_algorithm not in self._op_algo_cache:
                self._op_algo_cache[self.optimisation_algorithm] = desired_cls()
            self.op_algo_parameters = self._op_algo_cache[self.optimisation_algorithm]

    def _update_sim_metric_params(self):
        desired_cls: type = Parameters.SIM_MET_PARAM_CLASSES[self.sim_metric]
        if isinstance(self.sim_metric_parameters, desired_cls):
            self._sim_metric_cache[self.sim_metric] = self.sim_metric_parameters
        else:
            if self.sim_metric not in self._sim_metric_cache:
                self._sim_metric_cache[self.sim_metric] = desired_cls()
            self.sim_metric_parameters = self._sim_metric_cache[self.sim_metric]


class Context(HasTraits):
    parameters: Parameters = Instance(Parameters, allow_none=False)
    dadg: DirectedAcyclicDataGraph = Instance(DirectedAcyclicDataGraph, allow_none=False)
    namespace: str = Unicode(allow_none=True)
