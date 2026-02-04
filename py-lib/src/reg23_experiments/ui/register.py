import logging
from typing import Callable, Any
from abc import ABC, abstractmethod
import os

os.environ["QT_API"] = "PyQt6"

import torch
from magicgui import widgets
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from reg23_experiments.data.structs import Error
from reg23_experiments.ops.data_manager import DAG, ChildDAG
from reg23_experiments.ops.optimisation import mapping_transformation_to_parameters
from reg23_experiments.ui.viewer_singleton import viewer
from reg23_experiments.experiments.parameters import Parameters, PsoParameters
from reg23_experiments.ops.swarm import Swarm, OptimisationConfig as SwarmConfig

__all__ = ["RegisterGUI"]

logger = logging.getLogger(__name__)


class OptimisationAlgorithm(ABC):
    @abstractmethod
    def name(self) -> str:
        """
        :return: The name of this algorithm.
        """
        pass

    @abstractmethod
    def step(self) -> bool:
        """
        Execute one step of the optimisation
        :return: Whether the optimisation should terminate.
        """
        pass

    @abstractmethod
    def get_best(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_best_position(self) -> torch.Tensor:
        pass


class OptimisationJob:
    def __init__(self, *,  #
                 parent_dag: DAG,  #
                 max_iterations: int,  #
                 obj_func: Callable[[DAG | ChildDAG, torch.Tensor], torch.Tensor],  #
                 op_algo_cls: type,  #
                 **op_algo_kwargs):
        self._parent_dag = parent_dag
        self._max_iterations = max_iterations
        self._obj_func = obj_func
        self._op_algo_cls = op_algo_cls
        self._init_state: dict[str, Any] = op_algo_kwargs
        self._op_algo: OptimisationAlgorithm | None = None
        self._child_dag: ChildDAG | None = None
        self._it_count: int = 0

    @property
    def initialised(self) -> bool:
        return self._op_algo is not None

    @property
    def name(self) -> str:
        return self._op_algo.name()

    @property
    def best(self) -> torch.Tensor:
        return self._op_algo.get_best()

    @property
    def best_position(self) -> torch.Tensor:
        return self._op_algo.get_best_position()

    @property
    def iteration_count(self) -> int:
        return self._it_count

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, new_value: int) -> None:
        self._max_iterations = new_value

    def initialise(self) -> None:
        self._child_dag = ChildDAG(self._parent_dag)
        self._op_algo = self._op_algo_cls(  #
            objective_function=lambda x: self._obj_func(self._child_dag, x),  #
            **self._init_state  #
        )
        self._it_count = 0

    def step(self) -> bool:
        if self._it_count >= self._max_iterations:
            logger.warning(
                f"Optimisation job terminated instantly, as already reached the maximum of {self._max_iterations} "
                f"iterations.")
            return True
        self._it_count += 1
        return self._op_algo.step() or self._it_count >= self._max_iterations

    def reset(self) -> None:
        self.initialise()


class OptimisationWorker(QObject):
    progress = pyqtSignal(torch.Tensor, torch.Tensor)  # current best position, o.f. at position
    finished = pyqtSignal(torch.Tensor, torch.Tensor)  # converged position, o.f. value at position

    def __init__(self, optimisation_job: OptimisationJob):
        super().__init__()
        self._optimisation_job = optimisation_job

    def run(self):
        logger.info("Optimisation worker initialising...")
        logger.info("Optimisation worker initialised. Optimisation worker running...")
        self._optimisation_job.initialise()
        while not self._optimisation_job.step():
            self.progress.emit(self._optimisation_job.best_position, self._optimisation_job.best)
        logger.info("Optimisation worker finished.")
        self.finished.emit(self._optimisation_job.best_position, self._optimisation_job.best)


class PsoAlgorithm(OptimisationAlgorithm):
    def __init__(self, *, particle_count: int, starting_pos: torch.Tensor, starting_spread: float,
                 objective_function: Callable[[torch.Tensor], torch.Tensor], device):
        self._swarm = Swarm(config=SwarmConfig(objective_function=objective_function), dimensionality=6,
                            particle_count=particle_count, initialisation_position=starting_pos,
                            initialisation_spread=torch.full_like(starting_pos, starting_spread), device=device)

    def name(self) -> str:
        return "PSO"

    def step(self) -> bool:
        self._swarm.iterate()
        return False

    def get_best_position(self) -> torch.Tensor:
        return self._swarm.current_optimum_position

    def get_best(self) -> torch.Tensor:
        return self._swarm.current_optimum


def new_op_job(*, dag: DAG, parameters: Parameters,
               objective_function: Callable[[DAG | ChildDAG, torch.Tensor], torch.Tensor]) -> OptimisationJob:
    if parameters.optimisation_algorithm == "pso":
        assert isinstance(parameters.op_algo_parameters, PsoParameters)
        return OptimisationJob(  #
            parent_dag=dag,  #
            max_iterations=10,  #
            obj_func=objective_function,  #
            op_algo_cls=PsoAlgorithm,  #
            particle_count=parameters.op_algo_parameters.particle_count,  #
            starting_pos=mapping_transformation_to_parameters(dag.get("current_transformation")),  #
            starting_spread=1.0,  #
            device=dag.get("device"),  #
        )
    else:
        raise ValueError(f"Unrecognised optimisation algorithm: '{parameters.optimisation_algorithm}'.")


class RegisterGUI(widgets.Container):
    def __init__(self, *, parameters: Parameters, dag: DAG,
                 objective_function: Callable[[DAG | ChildDAG, torch.Tensor], torch.Tensor]):
        super().__init__(labels=False)
        self._parameters = parameters
        self._dag = dag
        self._objective_function = objective_function

        # -----
        # Evaluate once button and result
        # -----
        self._eval_once_button = widgets.PushButton(label="Evaluate once")
        self._eval_once_button.changed.connect(self._on_eval_once)
        self._eval_result_label = widgets.Label(label="Result:", value="n/a")
        self.append(widgets.Container(widgets=[  #
            self._eval_once_button,  #
            self._eval_result_label,  #
        ], layout="horizontal", label="Obj. func."))

        # -----
        # Run optimisation
        # -----
        self._op_job_cache: dict[str, OptimisationJob] = {}
        self._op_algo_state_display = widgets.Label()
        self.append(self._op_algo_state_display)
        self._update_op_algo_state_display()
        self._parameters.observe(lambda change: self._update_op_algo_state_display(), names=["optimisation_algorithm"])
        self._reset_state_button = widgets.PushButton(label="Reset state")
        self._reset_state_button.changed.connect(self._on_reset_state)
        self._one_iteration_button = widgets.PushButton(label="One iteration")
        self._one_iteration_button.changed.connect(self._on_one_iteration)
        self._state_label = widgets.Label()
        self.append(widgets.Container(widgets=[  #
            self._reset_state_button,  #
            self._one_iteration_button,  #
            self._state_label,  #
        ], layout="horizontal", label="Optimise"))

        # add self as widget in dock to the right
        viewer().window.add_dock_widget(self, name="Register", area="right", menu=viewer().window.window_menu,
                                        tabify=True)

    def _on_eval_once(self, *args) -> None:
        tr = self._dag.get("current_transformation")
        if isinstance(tr, Error):
            logger.error(f"Error getting 'current_transformation' for eval. once: {tr.description}")
            return
        self._eval_result_label.value = "{:.4f}".format(
            self._objective_function(self._dag, mapping_transformation_to_parameters(tr)).item())

    def _update_op_algo_state_display(self) -> None:
        if self._parameters.optimisation_algorithm not in self._op_job_cache:
            self._op_algo_state_display.value = "No algorithm initialised."
            return
        op_job_value = self._op_job_cache[self._parameters.optimisation_algorithm]
        if not op_job_value.initialised:
            self._op_algo_state_display.value = "No algorithm initialised."
            return
        self._op_algo_state_display.value = (
            "----- {} -----\n{} iterations have been performed out of {};\nCurrent best is f(x) = {:.4f}\nat x = {"
            "}").format(  #
            op_job_value.name,  #
            op_job_value.iteration_count,  #
            op_job_value.max_iterations,  #
            op_job_value.best,  #
            op_job_value.best_position,  #
        )

    def _on_reset_state(self, *args) -> None:
        if self._parameters.optimisation_algorithm in self._op_job_cache:
            self._op_job_cache.pop(self._parameters.optimisation_algorithm)
        self._op_job_cache[  #
            self._parameters.optimisation_algorithm] = new_op_job(  #
            dag=self._dag,  #
            parameters=self._parameters,  #
            objective_function=self._objective_function  #
        )
        self._update_op_algo_state_display()

    def _on_one_iteration(self, *args) -> None:
        if self._parameters.optimisation_algorithm not in self._op_job_cache:
            self._op_job_cache[self._parameters.optimisation_algorithm] = new_op_job(  #
                dag=self._dag,  #
                parameters=self._parameters,  #
                objective_function=self._objective_function  #
            )

        self._op_job_cache[  #
            self._parameters.optimisation_algorithm].max_iterations = self._op_job_cache[
                                                                          self._parameters.optimisation_algorithm].iteration_count + 1

        self._thread = QThread()
        self._worker = OptimisationWorker(self._op_job_cache[self._parameters.optimisation_algorithm])
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._update_op_algo_state_display)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._update_state_label_to_finished)
        # self._thread.finished.connect(self._thread_finish_callback)
        # self._worker.progress.connect(self._iteration_callback)
        self._thread.start()
        self._state_label.value = "Running..."

    def _update_state_label_to_finished(self) -> None:
        self._state_label.value = "Finished."
