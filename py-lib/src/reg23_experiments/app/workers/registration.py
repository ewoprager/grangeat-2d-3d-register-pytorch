from typing import Callable
import os
import logging

os.environ["QT_API"] = "PyQt6"

import torch
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from reg23_experiments.ops.optimisation import mapping_transformation_to_parameters
from reg23_experiments.data.structs import OptimisationInstance
from reg23_experiments.ops.optimisation_instances import PsoInstance
from reg23_experiments.app.state import AppState
from reg23_experiments.experiments.parameters import Parameters, PsoParameters
from reg23_experiments.ops.swarm import SwarmConfig
from reg23_experiments.ops.data_manager import DAG, ChildDAG

__all__ = ["new_optimisation_instance", "RegistrationWorker"]

logger = logging.getLogger(__name__)


def new_optimisation_instance(app_state: AppState, objective_function: Callable[
    [DAG | ChildDAG, torch.Tensor], torch.Tensor]) -> tuple[ChildDAG, OptimisationInstance]:
    child_dag = ChildDAG(app_state.dag)
    params = app_state.parameters
    if params.optimisation_algorithm == "pso":
        oa_params = params.op_algo_parameters
        assert isinstance(oa_params, PsoParameters)
        return child_dag, PsoInstance(  #
            particle_count=oa_params.particle_count,  #
            starting_pos=mapping_transformation_to_parameters(app_state.dag.get("current_transformation")),  #
            starting_spread=oa_params.starting_spread,  #
            config=SwarmConfig(  #
                objective_function=lambda x: objective_function(child_dag, x),  #
                inertia_coefficient=oa_params.inertia_coefficient,  #
                cognitive_coefficient=oa_params.cognitive_coefficient,  #
                social_coefficient=oa_params.social_coefficient,  #
            ),  #
            device=app_state.dag.get("device"),  #
        )
    # elif parameters.optimisation_algorithm == "local_search":
    #     pass
    else:
        raise ValueError(f"Unrecognised optimisation algorithm: '{params.optimisation_algorithm}'.")


class RegistrationWorker(QObject):
    progress = pyqtSignal(torch.Tensor, torch.Tensor)  # current best position, o.f. value at position
    finished = pyqtSignal(torch.Tensor, torch.Tensor)  # best position found, o.f. value at position

    def __init__(self, app_state: AppState, objective_function: Callable[
        [DAG | ChildDAG, torch.Tensor], torch.Tensor]):
        super().__init__()
        self._app_state = app_state
        self._objective_function = objective_function

    def run(self):
        logger.info("Optimisation worker initialising...")
        logger.info("Optimisation worker initialised. Optimisation worker running...")
        max_iterations = self._app_state.parameters.iteration_count
        child_dag, op_instance = new_optimisation_instance(self._app_state, self._objective_function)
        for it in range(max_iterations):
            terminate = op_instance.step()
            self.progress.emit(op_instance.get_best_position(), op_instance.get_best())
            if terminate:
                logger.info(f"Optimisation terminating after iteration {it + 1}/{max_iterations}.")
                break
        logger.info("Optimisation worker finished.")
        self.finished.emit(op_instance.get_best_position(), op_instance.get_best())
