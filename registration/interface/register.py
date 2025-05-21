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
from registration.interface.transformations import TransformationWidget
from registration.interface.lib.structs import *


def register(*, initial_transformation: Transformation, objective_function: Callable[[Transformation], torch.Tensor],
             iteration_callback: Callable[[torch.Tensor, torch.Tensor], None], particle_count: int,
             iteration_count: int) -> Transformation:
    def obj_func(params: torch.Tensor) -> torch.Tensor:
        return objective_function(Transformation(params[0:3], params[3:6]))

    logger.info("Optimising...")
    param_history = torch.zeros((particle_count * iteration_count, 6))
    value_history = torch.zeros(particle_count * iteration_count)
    start_params: torch.Tensor = initial_transformation.vectorised().cpu()

    if True:
        j: int = 0

        def objective_pso(particle_params: np.ndarray) -> np.ndarray:
            nonlocal j
            ret = np.zeros(particle_params.shape[0])
            for i, row in enumerate(particle_params):
                params = torch.tensor(copy.deepcopy(row))
                param_history[j] = params
                value = obj_func(params)
                value_history[j] = value
                j += 1
                iteration_callback(param_history[:j], value_history[:j])
                ret[i] = value.item()
            return ret

        options = {'c1': 2.525, 'c2': 1.225, 'w': 0.28}  # {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        n_dimensions = 6
        initial_positions = np.random.normal(loc=start_params.numpy(), size=(particle_count, n_dimensions),
                                             scale=np.array([0.1, 0.1, 0.1, 2.0, 2.0, 2.0]))
        initial_positions[0] = start_params.numpy()
        optimiser = pyswarms.single.GlobalBestPSO(n_particles=particle_count, dimensions=n_dimensions,
                                                  init_pos=initial_positions, options=options)

        cost, converged_params = optimiser.optimize(objective_pso, iters=iteration_count)
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
    progress = pyqtSignal(torch.Tensor, torch.Tensor)

    def __init__(self, *, initial_transformation: Transformation,
                 objective_function: Callable[[Transformation], torch.Tensor], particle_count: int,
                 iteration_count: int):
        super().__init__()
        self._initial_transformation = initial_transformation
        self._objective_function = objective_function
        self._particle_count = particle_count
        self._iteration_count = iteration_count

    def run(self):
        res = register(initial_transformation=self._initial_transformation, iteration_callback=self._iteration_callback,
                       objective_function=self._objective_function, particle_count=self._particle_count,
                       iteration_count=self._iteration_count)
        self.finished.emit(res)

    def _iteration_callback(self, param_history: torch.Tensor, value_history: torch.Tensor) -> None:
        self.progress.emit(param_history, value_history)


class RegisterWidget(widgets.Container):
    def __init__(self, *, transformation_widget: TransformationWidget,
                 objective_functions: dict[str, Callable[[Transformation], torch.Tensor]]):
        super().__init__()
        self._transformation_widget = transformation_widget
        self._objective_functions = objective_functions

        # Optimisation worker thread and plotting
        self._evals_per_render: int = 100
        self._iteration_callback_count: int = 0
        self._best: float | None = None
        self._last_rendered_iteration: int = 0
        self._thread: QThread | None = None
        self._worker: Worker | None = None

        # Optimisation hyper-parameters:
        self._particle_count: int = 500
        self._iteration_count: int = 5

        self._fig, self._axes = plt.subplots()
        self.native.layout().addWidget(FigureCanvasQTAgg(self._fig))

        self._objective_function_widget = WidgetSelectData(widget_type=widgets.ComboBox,
                                                           initial_choices=objective_functions, label="Obj. func.")

        self._eval_once_button = widgets.PushButton(label="Evaluate once")
        self._eval_once_button.changed.connect(self._on_eval_once)

        self._eval_result_label = widgets.Label(label="Result:", value="n/a")

        self.append(widgets.Container(
            widgets=[self._objective_function_widget.widget, self._eval_once_button, self._eval_result_label],
            layout="horizontal", label="Obj. func."))

        self._particle_count_widget = widgets.SpinBox(value=self._particle_count, min=1, max=5000, step=1,
                                                      label="N particles")
        self._particle_count_widget.changed.connect(self._on_particle_count)

        self._iteration_count_widget = widgets.SpinBox(value=self._iteration_count, min=1, max=30, step=1,
                                                       label="N iterations")
        self._iteration_count_widget.changed.connect(self._on_iteration_count)

        self.append(
            widgets.Container(widgets=[self._particle_count_widget, self._iteration_count_widget], layout="horizontal",
                              label="PSO params"))

        self._register_button = widgets.PushButton(label="Register")
        self._register_button.changed.connect(self._on_register)
        self.append(self._register_button)

        self._register_progress = widgets.Label(label="Progress:", value="n/a")
        self.append(self._register_progress)

        self._evals_per_render_widget = widgets.SpinBox(value=self._evals_per_render, min=1, max=1000, step=1,
                                                        label="Evals./re-plot")
        self._evals_per_render_widget.changed.connect(self._on_evals_per_render)
        self.append(self._evals_per_render_widget)

    def _current_obj_func(self) -> Callable[[Transformation], torch.Tensor] | None:
        current = self._objective_function_widget.get_selected()  # from a ComboBox, a str is returned
        return self._objective_function_widget.get_data(current)

    def _on_eval_once(self):
        res = self._current_obj_func()(self._transformation_widget.get_current_transformation())
        self._eval_result_label.value = "{:.4f}".format(res.item())

    def _on_register(self):
        self._iteration_callback_count = 0
        self._best = None
        self._last_rendered_iteration = 0
        self._thread = QThread()
        self._worker = Worker(initial_transformation=self._transformation_widget.get_current_transformation(),
                              objective_function=self._current_obj_func(), particle_count=self._particle_count,
                              iteration_count=self._iteration_count)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._finish_callback)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._worker.progress.connect(self._iteration_callback)
        self._thread.start()

    def _on_evals_per_render(self, *args):
        self._evals_per_render = self._evals_per_render_widget.get_value()

    def _iteration_callback(self, param_history: torch.Tensor, value_history: torch.Tensor) -> None:
        self._iteration_callback_count += 1

        if self._iteration_callback_count - self._last_rendered_iteration < self._evals_per_render:
            return

        self._axes.cla()

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
        self._axes.plot(its, value_history)
        self._axes.set_xlabel("iteration")
        self._axes.set_ylabel("objective function value")
        self._fig.canvas.draw()

        self._last_rendered_iteration = self._iteration_callback_count
        values = torch.tensor(value_history).cpu()
        self._best, best_index = values.min(0)
        self._best = self._best.item()
        best_index = best_index.item()
        best_transformation = Transformation(param_history[best_index][0:3], param_history[best_index][3:6])
        self._transformation_widget.set_current_transformation(best_transformation)

        self._register_progress.value = "Eval. {} / {}; best = {:.4f}".format(self._iteration_callback_count,
                                                                              self._particle_count *
                                                                              self._iteration_count,
                                                                              0.0 if self._best is None else self._best)

    def _finish_callback(self, converged_transformation: Transformation):
        self._transformation_widget.set_current_transformation(converged_transformation)
        self._transformation_widget.save_transformation(converged_transformation, "registration result {}".format(
            datetime.now().strftime("%Y-%m-%d, %H:%M:%S")))

    def _on_particle_count(self, *args) -> None:
        self._particle_count = self._particle_count_widget.get_value()

    def _on_iteration_count(self, *args) -> None:
        self._iteration_count = self._iteration_count_widget.get_value()
