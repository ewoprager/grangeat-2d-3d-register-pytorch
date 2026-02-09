from typing import Callable
import os
import logging

import traitlets

os.environ["QT_API"] = "PyQt6"

import torch
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from reg23_experiments.ops.optimisation import mapping_transformation_to_parameters
from reg23_experiments.data.structs import OptimisationInstance
from reg23_experiments.ops.optimisation_instances import PsoInstance
from reg23_experiments.app.state import AppState, WorkerState
from reg23_experiments.experiments.parameters import Parameters, PsoParameters
from reg23_experiments.ops.swarm import SwarmConfig
from reg23_experiments.experiments.parameters import Context
from reg23_experiments.ops.data_manager import DAG, ChildDAG
from reg23_experiments.utils.data import clone_has_traits

__all__ = ["new_optimisation_instance", "RegistrationWorker"]

logger = logging.getLogger(__name__)


def new_optimisation_instance(app_state: AppState,
                              objective_function: Callable[[Context, torch.Tensor], torch.Tensor]) -> tuple[
    Context, OptimisationInstance]:
    context = Context(parameters=clone_has_traits(app_state.parameters), dag=ChildDAG(app_state.dag))
    if context.parameters.optimisation_algorithm == "pso":
        oa_params = context.parameters.op_algo_parameters
        assert isinstance(oa_params, PsoParameters)
        return context, PsoInstance(  #
            particle_count=oa_params.particle_count,  #
            starting_pos=mapping_transformation_to_parameters(app_state.dag.get("current_transformation")),  #
            starting_spread=oa_params.starting_spread,  #
            config=SwarmConfig(  #
                objective_function=lambda x: objective_function(context, x),  #
                inertia_coefficient=oa_params.inertia_coefficient,  #
                cognitive_coefficient=oa_params.cognitive_coefficient,  #
                social_coefficient=oa_params.social_coefficient,  #
            ),  #
            device=app_state.dag.get("device"),  #
        )
    # elif parameters.optimisation_algorithm == "local_search":
    #     pass
    else:
        raise ValueError(f"Unrecognised optimisation algorithm: '{context.parameters.optimisation_algorithm}'.")


class RegistrationWorker(QObject):
    progress = pyqtSignal(WorkerState)  # current best position, o.f. value at position
    finished = pyqtSignal(WorkerState)  # best position found, o.f. value at position

    def __init__(self, *, app_state: AppState, objective_function: Callable[[Context, torch.Tensor], torch.Tensor],
                 max_iterations: int | None = None):
        super().__init__()
        self._max_iterations = max_iterations
        self._app_state = app_state
        self._objective_function = objective_function

        if self._max_iterations is None:
            self._max_iterations = self._app_state.parameters.iteration_count

    def run(self):
        logger.info("Optimisation worker initialising...")
        self.progress.emit(WorkerState(  #
            current_best_x=None,  #
            current_best_f=None,  #
            iteration="initialising",  #
            max_iterations=self._max_iterations,  #
        ))
        child_dag, op_instance = new_optimisation_instance(self._app_state, self._objective_function)
        logger.info(
            f"Optimisation worker initialised. Optimisation worker running for {self._max_iterations} iterations...")
        self.progress.emit(WorkerState(  #
            current_best_x=op_instance.get_best_position(),  #
            current_best_f=op_instance.get_best(),  #
            iteration=0,  #
            max_iterations=self._max_iterations,  #
        ))
        for it in range(self._max_iterations):
            terminate = op_instance.step()
            self.progress.emit(WorkerState(  #
                current_best_x=op_instance.get_best_position(),  #
                current_best_f=op_instance.get_best(),  #
                iteration=it + 1,  #
                max_iterations=self._max_iterations,  #
            ))
            if terminate:
                logger.info(f"Optimisation terminating after iteration {it + 1}/{self._max_iterations}.")
                break
        logger.info("Optimisation worker finished.")
        self.finished.emit(WorkerState(  #
            current_best_x=op_instance.get_best_position(),  #
            current_best_f=op_instance.get_best(),  #
            iteration="finished",  #
            max_iterations=self._max_iterations,  #
        ))
