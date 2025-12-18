import logging
from typing import Callable
from abc import ABC, abstractmethod

import torch
from magicgui import widgets

from reg23_experiments.program.lib.structs import Error
from reg23_experiments.program import data_manager
from reg23_experiments.program.modules.interface import viewer
from reg23_experiments.registration.interface.lib.structs import WidgetSelectData
from reg23_experiments.registration.lib.structs import Transformation
from reg23_experiments.program.lib import optimisation

__all__ = ["OpAlgoWidget", "PSOWidget", "RegisterGUI"]

logger = logging.getLogger(__name__)


class OpAlgoWidget(ABC):
    @abstractmethod
    def get_op_algo(self) -> Callable:
        pass


class PSOWidget(widgets.Container, OpAlgoWidget):
    def __init__(self):
        super().__init__(layout="horizontal")

        particle_count = data_manager().get("pso_particle_count")
        iteration_count = data_manager().get("pso_iteration_count")

        self._particle_count_widget = widgets.SpinBox(value=particle_count, min=1, max=5000, step=1,
                                                      label="N particles")
        self._particle_count_widget.changed.connect(self._on_particle_count)
        self.append(self._particle_count_widget)

        self._iteration_count_widget = widgets.SpinBox(value=iteration_count, min=1, max=30, step=1,
                                                       label="N iterations")
        self._iteration_count_widget.changed.connect(self._on_iteration_count)
        self.append(self._iteration_count_widget)

    def get_op_algo(self) -> Callable:
        # return ParticleSwarm(particle_count=self._particle_count_widget.get_value(),
        #                      iteration_count=self._iteration_count_widget.get_value())
        return optimisation.pso

    def set_particle_count(self, new_value: int) -> None:
        self._particle_count_widget.set_value(new_value)

    def set_iteration_count(self, new_value: int) -> None:
        self._iteration_count_widget.set_value(new_value)

    def _on_particle_count(self, _) -> None:
        data_manager().set_data("pso_particle_count", self._particle_count_widget.get_value())

    def _on_iteration_count(self, _) -> None:
        data_manager().set_data("pso_iteration_count", self._iteration_count_widget.get_value())


# class LocalSearchWidget(widgets.Container, OpAlgoWidget):
#     def __init__(self, *, no_improvement_threshold: int, max_reductions: int):
#         super().__init__(layout="vertical", labels=False)
#
#         self._no_improvement_threshold_widget = widgets.SpinBox(value=no_improvement_threshold, min=2, max=1000, step=1,
#                                                                 label="reduction thresh.")
#
#         self._max_reductions_widget = widgets.SpinBox(value=max_reductions, min=0, max=500, step=1,
#                                                       label="max reductions")
#         self.append(widgets.Container(widgets=[self._no_improvement_threshold_widget, self._max_reductions_widget],
#                                       layout="horizontal"))
#
#         self._reduction_ratio_widget = widgets.FloatSpinBox(value=.75, min=0.0, max=1.0, label="reduction ratio")
#
#         self.append(widgets.Container(widgets=[self._reduction_ratio_widget], layout="horizontal"))
#
#     def get_op_algo(self) -> LocalSearch:
#         return LocalSearch(no_improvement_threshold=self._no_improvement_threshold_widget.get_value(),
#                            max_reductions=self._max_reductions_widget.get_value(),
#                            reduction_ratio=self._reduction_ratio_widget.get_value())
#
#     def set_from_op_algo(self, local_search: LocalSearch) -> None:
#         self._no_improvement_threshold_widget.set_value(local_search.no_improvement_threshold)
#         self._max_reductions_widget.set_value(local_search.max_reductions)
#         self._reduction_ratio_widget.set_value(local_search.reduction_ratio)


class RegisterGUI(widgets.Container):
    def __init__(self, objective_functions: dict[str, Callable[[Transformation], torch.Tensor]]):
        super().__init__(labels=False)

        data_manager().set_data_multiple(pso_particle_count=2000, pso_iteration_count=10,
                                         objective_function=next(iter(objective_functions.items()))[1])

        ##
        ## Objective function
        ##
        self._objective_function_widget = WidgetSelectData(widget_type=widgets.ComboBox,
                                                           initial_choices=objective_functions, label="Obj. func.")
        self._objective_function_widget.widget.changed.connect(self._on_objective_function)

        self._eval_once_button = widgets.PushButton(label="Evaluate once")
        self._eval_once_button.changed.connect(self._on_eval_once)

        self._eval_result_label = widgets.Label(label="Result:", value="n/a")

        self.append(widgets.Container(
            widgets=[self._objective_function_widget.widget, self._eval_once_button, self._eval_result_label],
            layout="horizontal", label="Obj. func."))

        ##
        ## Optimisation algorithm and parameters
        ##
        self._op_algo_widgets = {"PSO": PSOWidget(),
                                 # LocalSearch.algorithm_name(): LocalSearchWidget(no_improvement_threshold=10,
                                 #                                                 max_reductions=4)
                                 }
        self._algorithm_widget = widgets.ComboBox(choices=[name for name in self._op_algo_widgets])
        self._algorithm_widget.changed.connect(self._on_algorithm)
        self._algorithm_container_widget = widgets.Container(widgets=[self._algorithm_widget], layout="vertical")
        self.append(self._algorithm_container_widget)
        self._refresh_algorithm_container_widget()

        viewer().window.add_dock_widget(self, name="Register", area="right", menu=viewer().window.window_menu,
                                        tabify=True)

    def _on_eval_once(self, *args) -> None:
        of = data_manager().get("objective_function")
        if isinstance(of, Error):
            logger.error(f"Error getting 'objective_function' for eval. once: {of.description}")
            return
        tr = data_manager().get("current_transformation")
        if isinstance(tr, Error):
            logger.error(f"Error getting 'current_transformation' for eval. once: {of.description}")
            return
        self._eval_result_label.value = of(transformation=tr).item()

    def _on_algorithm(self, *args) -> None:
        self._refresh_algorithm_container_widget()

    def _on_objective_function(self, **args) -> None:
        current = self._objective_function_widget.get_selected()  # from a ComboBox, a str is returned
        data_manager().set_data("objective_function", self._objective_function_widget.get_data(current))

    def _refresh_algorithm_container_widget(self) -> None:
        if len(self._algorithm_container_widget) > 1:
            del self._algorithm_container_widget[-1]

        value = self._algorithm_widget.get_value()
        if value in self._op_algo_widgets:
            self._algorithm_container_widget.append(self._op_algo_widgets[value])
            data_manager().set_data("optimisation_algorithm", self._op_algo_widgets[value].get_op_algo())
        else:
            logger.error("Unrecognised optimisation algorithm option: '{}'.".format(value))
