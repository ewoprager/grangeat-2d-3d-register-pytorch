import logging
from typing import Callable

import torch
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from reg23_experiments.app.state import AppState
from reg23_experiments.ops.data_manager import DAG, ChildDAG
from reg23_experiments.ops.optimisation import mapping_transformation_to_parameters
from reg23_experiments.app.workers.registration import RegistrationWorker

__all__ = ["WorkerManager"]

logger = logging.getLogger(__name__)


class WorkerManager:
    def __init__(self, *, app_state: AppState, objective_function: Callable[
        [DAG | ChildDAG, torch.Tensor], torch.Tensor]):
        self._app_state = app_state
        self._objective_function = objective_function

        self._app_state.observe(self._button_evaluate_once, names=["button_evaluate_once"])
        self._app_state.observe(self._button_run_one_iteration, names=["button_run_one_iteration"])

        self._thread = None
        self._worker = None

    def _button_evaluate_once(self, change) -> None:
        if not change.new:
            return
        self._app_state.button_evaluate_once = False

        result = self._objective_function(self._app_state.dag, mapping_transformation_to_parameters(
            self._app_state.dag.get("current_transformation")))
        self._app_state.eval_once_result = "{:.4f}".format(result.item())

    def _button_run_one_iteration(self, change) -> None:
        if not change.new:
            return
        self._app_state.button_run_one_iteration = False

        self._thread = QThread()
        self._worker = RegistrationWorker(app_state=self._app_state, objective_function=self._objective_function)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._update_job_state_description_label)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        # self._thread.finished.connect(self._update_state_label_to_finished)
        # self._thread.finished.connect(self._thread_finish_callback)
        # self._worker.progress.connect(self._iteration_callback)
        self._thread.start()
        # self._state_label.value = "Running..."

    def _update_job_state_description_label(self, best_position: torch.Tensor, best: torch.Tensor) -> None:
        self._app_state.job_state_description = "Current best is f(x) = {:.4f}\nat x = {}".format(best.item(),
                                                                                                  best_position)
