from traitlets import HasTraits, Int, Float, Instance, Bool, Enum, Unicode, Undefined, observe

from reg23_experiments.utils.data import StrictHasTraits


class PsoParameters(StrictHasTraits):
    inertia_coefficient = Float(min=0.0, default_value=0.28).tag(ui=True)
    cognitive_coefficient = Float(min=0.0, default_value=2.525).tag(ui=True)
    social_coefficient = Float(min=0.0, default_value=1.225).tag(ui=True)


class LocalSearchParameters(StrictHasTraits):
    initial_step_size = Float(min=0.0, default_value=0.1).tag(ui=True)
    no_improvement_threshold = Int(min=0, default_value=10).tag(ui=True)
    step_size_reduction_ratio = Float(min=0.0, max=1.0, default_value=0.75).tag(ui=True)
    max_reductions = Int(min=0, default_value=4).tag(ui=True)
    max_iterations = Int(min=1, default_value=5000).tag(ui=True)


class Parameters(StrictHasTraits):
    ct_path = Unicode(default_value=Undefined).tag(ui=True)
    downsample_level = Int(min=0, default_value=Undefined).tag(ui=True)
    truncation_percent = Int(min=0, max=100, default_value=Undefined).tag(ui=True)
    cropping = Enum(values=[  #
        "None",  #
        "nonzero_drr",  #
        "full_depth_drr"  #
    ]).tag(ui=True)
    mask = Enum(values=[  #
        "None",  #
        "Every evaluation",  #
        "Every evaluation weighting zncc"  #
    ], default_value=Undefined).tag(ui=True)
    sim_metric = Enum(values=[  #
        "zncc",  #
        "local_zncc",  #
        "multiscale_zncc",  #
        "gradient_correlation"  #
    ], default_value=Undefined).tag(ui=True)
    starting_distance = Float(min=0.0)
    sample_count_per_distance = Int(min=1)
    optimisation_algorithm = Enum(values=[  #
        "pso",  #
        "local_search"  #
    ], default=Undefined).tag(ui=True)
    op_algo_parameters = Instance(HasTraits, allow_none=False, default_value=Undefined).tag(ui=True)

    def __init__(self, **kwargs):
        self._op_algo_cache = {}
        super().__init__(**kwargs)
        self._update_op_algo_params()

    @observe("optimisation_algorithm")
    def _op_algo_changed(self, _):
        self._update_op_algo_params()

    def _update_op_algo_params(self):
        if self.optimisation_algorithm not in self._op_algo_cache:
            if self.optimisation_algorithm == "pso":
                self._op_algo_cache[self.optimisation_algorithm] = PsoParameters()
            elif self.optimisation_algorithm == "local_search":
                self._op_algo_cache[self.optimisation_algorithm] = LocalSearchParameters()
            else:
                raise ValueError(f"Unrecognised optimisation algorithm '{self.optimisation_algorithm}'.")

        self.op_algo_parameters = self._op_algo_cache[self.optimisation_algorithm]
