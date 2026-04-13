import logging
from typing import Callable

import torch
from PyQt6.QtCore import QThread

from reg23_experiments.app.context import AppContext
from reg23_experiments.app.state import WorkerState
from reg23_experiments.app.workers.registration_worker import RegistrationWorker
from reg23_experiments.experiments.parameters import Context
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation, \
    mapping_transformation_to_parameters

__all__ = ["WorkerManager"]

logger = logging.getLogger(__name__)


class WorkerManager:
    """
    No GUI / widgets

    Reads from and write to the state and DADG
    """

    def __init__(self, *, ctx: AppContext, objective_function: Callable[[Context, torch.Tensor], torch.Tensor]):
        self._ctx = ctx
        self._objective_function = objective_function

        self._ctx.state.observe(self._button_evaluate_once, names=["button_evaluate_once"])
        self._ctx.state.observe(self._button_run_one_iteration, names=["button_run_one_iteration"])
        self._ctx.state.observe(self._button_run, names=["button_run"])
        self._ctx.state.observe(self._button_load_current_best, names=["button_load_current_best"])

        self._thread = None
        self._worker = None

    @property
    def _xray_selected(self) -> bool:
        return self._ctx.state.register_xray_choice is not None

    @property
    def _c_t_key(self) -> str:
        return f"{self._ctx.state.register_xray_choice}__current_transformation"

    def _button_evaluate_once(self, change) -> None:
        if not change.new:
            return
        self._ctx.state.button_evaluate_once = False
        #
        if not self._xray_selected:
            logger.warning("Cannot evaluate objective function: no X-ray selected.")
            return
        context = Context(parameters=self._ctx.state.parameters, dadg=self._ctx.dadg,
                          namespace=self._ctx.state.register_xray_choice)
        result = self._objective_function(context,
                                          mapping_transformation_to_parameters(self._ctx.dadg.get(self._c_t_key)))
        self._ctx.state.eval_once_result = "{:.4f}".format(result.item())

    def _button_run_one_iteration(self, change) -> None:
        if not change.new:
            return
        self._ctx.state.button_run_one_iteration = False
        #
        if not self._xray_selected:
            logger.warning("Cannot run registration: no X-ray selected.")
            return
        self._thread = QThread()
        self._worker = RegistrationWorker(ctx=self._ctx, objective_function=self._objective_function, max_iterations=1)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._update_from_worker)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        # self._thread.finished.connect(self._update_state_label_to_finished)
        # self._thread.finished.connect(self._thread_finish_callback)
        # self._worker.progress.connect(self._iteration_callback)
        self._thread.start()  # self._state_label.value = "Running..."

    def _button_run(self, change) -> None:
        if not change.new:
            return
        self._ctx.state.button_run = False
        #
        if not self._xray_selected:
            logger.warning("Cannot run registration: no X-ray selected.")
            return
        self._thread = QThread()
        self._worker = RegistrationWorker(ctx=self._ctx, objective_function=self._objective_function)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._update_from_worker)
        self._worker.finished.connect(self._update_from_worker)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        # self._thread.finished.connect(self._update_state_label_to_finished)
        # self._thread.finished.connect(self._thread_finish_callback)
        # self._worker.progress.connect(self._iteration_callback)
        self._thread.start()  # self._state_label.value = "Running..."

    def _update_from_worker(self, worker_state: WorkerState) -> None:
        self._ctx.state.worker_state = worker_state

    def _button_load_current_best(self, change) -> None:
        if not change.new:
            return
        self._ctx.state.button_load_current_best = False
        #
        if not self._xray_selected:
            logger.warning("Cannot load current best: no X-ray selected.")
            return
        if self._ctx.state.worker_state is None:
            logger.warning("Cannot load current best: no registration has been run.")
            return
        self._ctx.dadg.set(self._c_t_key,
                           mapping_parameters_to_transformation(self._ctx.state.worker_state.current_best_x))
