import copy
import time
import logging
from datetime import datetime
from typing import Callable, Any

logger = logging.getLogger(__name__)

import torch
import napari
from magicgui import magicgui, widgets
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import pyswarms

from registration.lib.structs import *
from registration.interface.transformations import TransformationManager


def register(*, initial_transformation: Transformation, objective_function: Callable[[Transformation], torch.Tensor],
             iteration_callback: Callable[[list, list], None]) -> Transformation:
    def obj_func(params: torch.Tensor) -> torch.Tensor:
        return objective_function(Transformation(params[0:3], params[3:6]))

    logger.info("Optimising...")
    param_history = []
    value_history = []
    start_params: torch.Tensor = initial_transformation.vectorised().cpu()

    if True:
        def objective_pso(particle_params: np.ndarray) -> np.ndarray:
            ret = np.zeros(particle_params.shape[0])
            for i, row in enumerate(particle_params):
                params = torch.tensor(copy.deepcopy(row))
                param_history.append(params)
                value = obj_func(params)
                value_history.append(value)
                iteration_callback(param_history, value_history)
                ret[i] = value.item()
            return ret

        options = {'c1': 2.525, 'c2': 1.225, 'w': 0.28}  # {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        n_particles = 500
        n_dimensions = 6
        optimiser = pyswarms.single.GlobalBestPSO(n_particles=n_particles, dimensions=n_dimensions,
                                                  init_pos=np.random.normal(loc=start_params.numpy(),
                                                                            size=(n_particles, n_dimensions),
                                                                            scale=np.array(
                                                                                [0.1, 0.1, 0.1, 2.0, 2.0, 2.0])),
                                                  options=options)

        cost, converged_params = optimiser.optimize(objective_pso, iters=10)
        converged_params = torch.from_numpy(converged_params)
        logger.info("PSO finished")

    if False:
        def objective_scipy(params: np.ndarray) -> float:
            params = torch.tensor(copy.deepcopy(params))
            param_history.append(params)
            value = obj_func(params)
            value_history.append(value)
            iteration_callback(param_history, value_history)
            return value.item()

        tic = time.time()
        # res = scipy.optimize.minimize(objective_scipy, start_params, method='Powell')
        res = scipy.optimize.basinhopping(objective_scipy, start_params, T=1.0,
                                          minimizer_kwargs={"method": 'Nelder-Mead'})
        toc = time.time()
        logger.info("Done. Took {:.3f}s.".format(toc - tic))
        logger.info(res)
        converged_params = torch.from_numpy(res.x)

    return Transformation(converged_params[0:3], converged_params[3:6])


class Worker(QObject):
    finished = pyqtSignal(Transformation)
    progress = pyqtSignal(list, list)

    def __init__(self, *, initial_transformation: Transformation,
                 objective_function: Callable[[Transformation], torch.Tensor]):
        super().__init__()
        self.initial_transformation = initial_transformation
        self.objective_function = objective_function

    def run(self):
        def iteration_callback(param_history: list, value_history: list) -> None:
            nonlocal self
            self.progress.emit(param_history, value_history)

        res = register(initial_transformation=self.initial_transformation, iteration_callback=iteration_callback,
                       objective_function=self.objective_function)
        self.finished.emit(res)


class RegisterWidget(widgets.Container):
    def __init__(self, *, transformation_manager: TransformationManager,
                 objective_function: Callable[[Transformation], torch.Tensor]):
        super().__init__()
        self.transformation_manager = transformation_manager
        self.objective_function = objective_function

        self.fig, self.axes = plt.subplots()

        eval_once_button = widgets.PushButton(label="Evaluate once")
        self.eval_result_label = widgets.Label(label="Evaluation result: -")

        @eval_once_button.changed.connect
        def _():
            nonlocal self
            res = objective_function(self.transformation_manager.get_current_transformation())
            self.eval_result_label.label = "Evaluation result: {:.4f}".format(res.item())

        register_button = widgets.PushButton(label="Register")

        self.register_progress = widgets.Label(label="no registration run")
        self.evals_per_render = 10
        self.iteration_callback_count = 0
        self.best = None
        self.last_value = None
        self.last_rendered_iteration = 0

        self.evals_per_render_widget = widgets.SpinBox(value=10, min=1, max=1000, step=1,
                                                       label="Evaluations per re-plot")

        @self.evals_per_render_widget.changed.connect
        def _(new_value):
            nonlocal self
            self.evals_per_render = new_value

        def iteration_callback(param_history: list, value_history: list) -> None:
            nonlocal self
            self.iteration_callback_count += 1
            self.last_value = value_history[-1].item()
            if self.iteration_callback_count - self.last_rendered_iteration >= self.evals_per_render:
                self.axes.cla()
                param_history = torch.stack(param_history, dim=0)
                value_history = torch.tensor(value_history)

                its = np.arange(param_history.size()[0])
                its2 = np.array([its[0], its[-1]])

                # rotations
                # self.axes.plot(its2, np.full(2, 0.), ls='dashed')
                # self.axes.plot(its, param_history[:, 0], label="r0")
                # self.axes.plot(its, param_history[:, 1], label="r1")
                # self.axes.plot(its, param_history[:, 2], label="r2")
                # self.axes.legend()
                # self.axes.set_xlabel("iteration")
                # self.axes.set_ylabel("param value [rad]")
                # self.axes.set_title("rotation parameter values over optimisation iterations")
                # self.fig.canvas.draw()

                # value
                self.axes.plot(its, value_history)
                self.axes.set_xlabel("iteration")
                self.axes.set_ylabel("objective function value")
                self.fig.canvas.draw()

                self.last_rendered_iteration = self.iteration_callback_count
                values = torch.tensor(value_history).cpu()
                self.best, best_index = values.min(0)
                self.best = self.best.item()
                best_index = best_index.item()
                best_transformation = Transformation(param_history[best_index][0:3], param_history[best_index][3:6])
                self.transformation_manager.set_transformation(best_transformation)

            self.register_progress.label = "Iteration {}: latest = {:.4f}, best = {:.4f}".format(
                self.iteration_callback_count, self.last_value, 0.0 if self.best is None else self.best)

        def finish_callback(converged_transformation: Transformation):
            nonlocal self
            self.transformation_manager.set_transformation(converged_transformation)
            self.transformation_manager.save_transformation(converged_transformation, "registration result {}".format(
                datetime.now().strftime("%Y-%m-%d, %H:%M:%S")))

        @register_button.changed.connect
        def _():
            nonlocal self
            self.iteration_callback_count = 0
            self.best = None
            self.last_value = None
            self.last_rendered_iteration = 0
            self.thread = QThread()
            self.worker = Worker(initial_transformation=self.transformation_manager.get_current_transformation(),
                                 objective_function=self.objective_function)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(finish_callback)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(iteration_callback)
            self.thread.start()

        self.native.layout().addWidget(FigureCanvasQTAgg(self.fig))
        self.append(eval_once_button)
        self.append(self.eval_result_label)
        self.append(register_button)
        self.append(self.register_progress)
        self.append(self.evals_per_render_widget)
