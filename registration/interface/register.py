import copy
import time
import logging
from datetime import datetime
from typing import Callable, Any, NamedTuple, Union
from abc import ABC, abstractmethod
from enum import Enum

import scipy.optimize

logger = logging.getLogger(__name__)

import torch
import napari
from magicgui import magicgui, widgets
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import pyswarms
from qtpy.QtWidgets import QApplication

from registration.lib.structs import *
from registration.interface.transformations import TransformationWidget
from registration.interface.lib.structs import *


class OptimisationAlgorithm(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def algorithm_name() -> str:
        pass

    @abstractmethod
    def run(self, *, starting_parameters: torch.Tensor, objective_function: Callable[[torch.Tensor], torch.Tensor],
            iteration_callback: Callable[[torch.Tensor, torch.Tensor], None]) -> torch.Tensor:
        pass


class ParticleSwarm(OptimisationAlgorithm):
    def __init__(self, particle_count: int, iteration_count: int):
        self._particle_count = particle_count
        self._iteration_count = iteration_count

    @property
    def particle_count(self):
        return self._particle_count

    @property
    def iteration_count(self):
        return self._iteration_count

    def __str__(self) -> str:
        return "Particle swarm with {} particles for {} iterations".format(self.particle_count, self.iteration_count)

    @staticmethod
    def algorithm_name() -> str:
        return "Particle Swarm"

    def run(self, *, starting_parameters: torch.Tensor, objective_function: Callable[[torch.Tensor], torch.Tensor],
            iteration_callback: Callable[[torch.Tensor, torch.Tensor], None]) -> torch.Tensor:
        n_dimensions = starting_parameters.numel()
        param_history = GrowingTensor([n_dimensions], self.particle_count * self.iteration_count)
        value_history = GrowingTensor([], self.particle_count * self.iteration_count)

        param_history.push_back(starting_parameters)
        value_history.push_back(objective_function(starting_parameters))

        def objective_pso(particle_params: np.ndarray) -> np.ndarray:
            ret = np.zeros(particle_params.shape[0])
            for i, row in enumerate(particle_params):
                params = torch.tensor(copy.deepcopy(row))
                param_history.push_back(params)
                value = objective_function(params)
                value_history.push_back(value)
                iteration_callback(param_history.get(), value_history.get())
                ret[i] = value.item()
            return ret

        options = {'c1': 2.525, 'c2': 1.225, 'w': 0.28}  # {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        initial_positions = np.random.normal(
            loc=starting_parameters.numpy(), size=(self.particle_count, n_dimensions),
            scale=np.array([0.1, 0.1, 0.1, 2.0, 2.0, 2.0]))
        initial_positions[0] = starting_parameters.numpy()
        optimiser = pyswarms.single.GlobalBestPSO(
            n_particles=self.particle_count, dimensions=n_dimensions, init_pos=initial_positions, options=options)

        cost, converged_params = optimiser.optimize(objective_pso, iters=self.iteration_count)
        return torch.from_numpy(converged_params)


class LocalSearch(OptimisationAlgorithm):
    def __init__(self):
        pass

    def __str__(self) -> str:
        return "Local search"

    @staticmethod
    def algorithm_name() -> str:
        return "Local Search"

    def run(self, *, starting_parameters: torch.Tensor, objective_function: Callable[[torch.Tensor], torch.Tensor],
            iteration_callback: Callable[[torch.Tensor, torch.Tensor], None]) -> torch.Tensor:
        n_dimensions = starting_parameters.numel()
        param_history = GrowingTensor([n_dimensions], 50)
        value_history = GrowingTensor([], 50)

        param_history.push_back(starting_parameters)
        value_history.push_back(objective_function(starting_parameters))

        def objective_scipy(params: np.ndarray) -> float:
            params = torch.tensor(copy.deepcopy(params))
            param_history.push_back(params)
            value = objective_function(params)
            value_history.push_back(value)
            iteration_callback(param_history.get(), value_history.get())
            return value.item()

        tic = time.time()
        res = scipy.optimize.minimize(objective_scipy, starting_parameters, method='Powell')
        toc = time.time()
        logger.info("Done. Took {:.3f}s.".format(toc - tic))
        logger.info(res)
        return torch.from_numpy(res.x)


class Worker(QObject):
    finished = pyqtSignal(Transformation)
    progress = pyqtSignal(torch.Tensor, torch.Tensor)

    def __init__(self, *, initial_transformation: Transformation,
                 objective_function: Callable[[Transformation], torch.Tensor], optimisation_algorithm: Any):
        super().__init__()
        self._initial_transformation = initial_transformation
        self._objective_function = objective_function
        self._optimisation_algorithm = optimisation_algorithm

    def run(self):
        def obj_func(params: torch.Tensor) -> torch.Tensor:
            return self._objective_function(Transformation.from_vector(params))

        starting_parameters: torch.Tensor = self._initial_transformation.vectorised().cpu()

        logger.info("Optimising with '{}'...".format(str(self._optimisation_algorithm)))
        converged_params = self._optimisation_algorithm.run(
            starting_parameters=starting_parameters, objective_function=obj_func,
            iteration_callback=self._iteration_callback)
        logger.info("Optimisation finished.")
        res = Transformation.from_vector(converged_params)
        self.finished.emit(res)

    def _iteration_callback(self, param_history: torch.Tensor, value_history: torch.Tensor) -> None:
        self.progress.emit(param_history, value_history)


class OpAlgoWidget(ABC):
    @abstractmethod
    def get_op_algo(self) -> OptimisationAlgorithm:
        pass

    @abstractmethod
    def set_from_op_algo(self, _: Any) -> None:
        pass


class PSOWidget(widgets.Container, OpAlgoWidget):
    def __init__(self, particle_count: int, iteration_count: int):
        super().__init__(layout="horizontal")
        self._particle_count = particle_count
        self._iteration_count = iteration_count

        self._particle_count_widget = widgets.SpinBox(
            value=self._particle_count, min=1, max=5000, step=1, label="N particles")
        self._particle_count_widget.changed.connect(self._on_particle_count)
        self.append(self._particle_count_widget)

        self._iteration_count_widget = widgets.SpinBox(
            value=self._iteration_count, min=1, max=30, step=1, label="N iterations")
        self._iteration_count_widget.changed.connect(self._on_iteration_count)
        self.append(self._iteration_count_widget)

    def get_op_algo(self) -> ParticleSwarm:
        return ParticleSwarm(particle_count=self._particle_count, iteration_count=self._iteration_count)

    def set_from_op_algo(self, particle_swarm: ParticleSwarm) -> None:
        self._particle_count_widget.set_value(particle_swarm.particle_count)
        self._iteration_count_widget.set_value(particle_swarm.iteration_count)

    def _on_particle_count(self, *args) -> None:
        self._particle_count = self._particle_count_widget.get_value()

    def _on_iteration_count(self, *args) -> None:
        self._iteration_count = self._iteration_count_widget.get_value()


class LocalSearchWidget(widgets.Container, OpAlgoWidget):
    def __init__(self):
        super().__init__(layout="horizontal")

    def get_op_algo(self) -> LocalSearch:
        return LocalSearch()

    def set_from_op_algo(self, local_search: LocalSearch) -> None:
        pass


class RegisterWidget(widgets.Container):
    def __init__(self, *, transformation_widget: TransformationWidget,
                 objective_functions: dict[str, Callable[[Transformation], torch.Tensor]],
                 fixed_image_crop_callback: Callable[[Cropping], None], hyper_parameter_save_path: str | pathlib.Path,
                 fixed_image_size: torch.Size):
        super().__init__(labels=False)
        self._transformation_widget = transformation_widget
        self._objective_functions = objective_functions
        self._hyper_parameter_save_path = pathlib.Path(hyper_parameter_save_path)

        # Optimisation worker thread and plotting
        self._evals_per_render: int = 100
        self._iteration_callback_count: int = 0
        self._best: float | None = None
        self._last_rendered_iteration: int = 0
        self._thread: QThread | None = None
        self._worker: Worker | None = None

        self._fig, self._axes = plt.subplots()
        self.native.layout().addWidget(FigureCanvasQTAgg(self._fig))

        ##
        ## Cropping sliders
        ##
        self._fixed_image_crop_callback = fixed_image_crop_callback
        self._ignore_crop_sliders: bool = False
        width = fixed_image_size[1]
        height = fixed_image_size[0]
        self._bottom_crop_slider = widgets.IntSlider(value=height, min=0, max=height, step=1, label="Crop bottom")
        self._top_crop_slider = widgets.IntSlider(value=0, min=0, max=height, step=1, label="Crop top")
        self._right_crop_slider = widgets.IntSlider(value=width, min=0, max=width, step=1, label="Crop right")
        self._left_crop_slider = widgets.IntSlider(value=0, min=0, max=width, step=1, label="Crop left")
        self._bottom_crop_slider.changed.connect(self._on_crop_slider)
        self._top_crop_slider.changed.connect(self._on_crop_slider)
        self._right_crop_slider.changed.connect(self._on_crop_slider)
        self._left_crop_slider.changed.connect(self._on_crop_slider)
        self.append(
            widgets.Container(
                widgets=[
                    self._bottom_crop_slider, self._top_crop_slider, self._right_crop_slider, self._left_crop_slider],
                layout="vertical"))

        ##
        ## Hyper-parameter saving and loading
        ##
        self._hyper_parameters_widget = WidgetManageSaved(
            initial_choices={"initial": self._current_hyper_parameters()}, DataType=HyperParameters,
            load_from_file=self._hyper_parameter_save_path, get_current_callback=self._current_hyper_parameters,
            set_callback=self._set_hyper_parameters)
        self.append(self._hyper_parameters_widget)

        ##
        ## Objective function
        ##
        self._objective_function_widget = WidgetSelectData(
            widget_type=widgets.ComboBox, initial_choices=objective_functions, label="Obj. func.")

        self._eval_once_button = widgets.PushButton(label="Evaluate once")
        self._eval_once_button.changed.connect(self._on_eval_once)

        self._eval_result_label = widgets.Label(label="Result:", value="n/a")

        self.append(
            widgets.Container(
                widgets=[self._objective_function_widget.widget, self._eval_once_button, self._eval_result_label],
                layout="horizontal", label="Obj. func."))

        ##
        ## Optimisation algorithm and parameters
        ##
        self._op_algo_widgets = {
            ParticleSwarm.algorithm_name(): PSOWidget(particle_count=500, iteration_count=5),
            LocalSearch.algorithm_name(): LocalSearchWidget()}
        self._algorithm_widget = widgets.ComboBox(choices=[name for name in self._op_algo_widgets])
        self._algorithm_widget.changed.connect(self._on_algorithm)
        self._algorithm_container_widget = widgets.Container(widgets=[self._algorithm_widget], layout="vertical")
        self.append(self._algorithm_container_widget)
        self._refresh_algorithm_container_widget()

        ##
        ## Registration
        ##
        self._register_button = widgets.PushButton(label="Register")
        self._register_button.changed.connect(self._on_register)

        self._register_progress = widgets.Label(label="Progress:", value="n/a")

        self._evals_per_render_widget = widgets.SpinBox(
            value=self._evals_per_render, min=1, max=1000, step=1, label="Evals./re-plot")
        self._evals_per_render_widget.changed.connect(self._on_evals_per_render)

        self.append(
            widgets.Container(
                widgets=[self._register_button, self._register_progress, self._evals_per_render_widget],
                layout="vertical"))

        QApplication.instance().aboutToQuit.connect(self._on_exit)

    def _current_obj_func(self) -> Callable[[Transformation], torch.Tensor] | None:
        current = self._objective_function_widget.get_selected()  # from a ComboBox, a str is returned
        return self._objective_function_widget.get_data(current)

    def _on_eval_once(self):
        res = self._current_obj_func()(self._transformation_widget.get_current_transformation())
        self._eval_result_label.value = "{:.4f}".format(res.item())

    def _on_register(self):
        if len(self._algorithm_container_widget) < 2:
            logger.warning("No optimisation algorithm selected.")
            return
        self._iteration_callback_count = 0
        self._best = None
        self._last_rendered_iteration = 0
        self._thread = QThread()
        self._worker = Worker(
            initial_transformation=self._transformation_widget.get_current_transformation(),
            objective_function=self._current_obj_func(),
            optimisation_algorithm=self._algorithm_container_widget[1].get_op_algo())
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
        values = value_history.cpu()
        self._best, best_index = values.min(0)
        self._best = self._best.item()
        best_index = best_index.item()
        best_transformation = Transformation(param_history[best_index][0:3], param_history[best_index][3:6])
        self._transformation_widget.set_current_transformation(best_transformation)

        self._register_progress.value = "{} evaluations, best = {:.4f}".format(
            self._iteration_callback_count, 0.0 if self._best is None else self._best)

    def _finish_callback(self, converged_transformation: Transformation):
        self._transformation_widget.set_current_transformation(converged_transformation)
        self._transformation_widget.save_transformation(
            converged_transformation, "registration result {}".format(
                datetime.now().strftime("%Y-%m-%d, %H:%M:%S")))

    def _on_algorithm(self, *args) -> None:
        self._refresh_algorithm_container_widget()

    def _refresh_algorithm_container_widget(self) -> None:
        if len(self._algorithm_container_widget) > 1:
            del self._algorithm_container_widget[-1]

        value = self._algorithm_widget.get_value()
        if value in self._op_algo_widgets:
            self._algorithm_container_widget.append(self._op_algo_widgets[value])
        else:
            logger.error("Unrecognised optimisation algorithm option: '{}'.".format(value))

    def _on_crop_slider(self, *args) -> None:
        if self._ignore_crop_sliders:
            return
        self._ignore_crop_sliders = True
        self._bottom_crop_slider.min = self._top_crop_slider.get_value() + 1
        self._top_crop_slider.max = self._bottom_crop_slider.get_value() - 1
        self._right_crop_slider.min = self._left_crop_slider.get_value() + 1
        self._left_crop_slider.max = self._right_crop_slider.get_value() - 1
        self._fixed_image_crop_callback(
            Cropping(
                top=self._top_crop_slider.get_value(), bottom=self._bottom_crop_slider.get_value(),
                left=self._left_crop_slider.get_value(), right=self._right_crop_slider.get_value()))
        self._ignore_crop_sliders = False

    def _current_hyper_parameters(self) -> HyperParameters:
        return HyperParameters(
            cropping=Cropping(
                right=self._right_crop_slider.get_value(), top=self._top_crop_slider.get_value(),
                left=self._left_crop_slider.get_value(), bottom=self._bottom_crop_slider.get_value()),
            source_offset=torch.zeros(2))

    def _set_hyper_parameters(self, new_value: HyperParameters) -> None:
        self._right_crop_slider.set_value(new_value.cropping.right)
        self._top_crop_slider.set_value(new_value.cropping.top)
        self._left_crop_slider.set_value(new_value.cropping.left)
        self._bottom_crop_slider.set_value(new_value.cropping.bottom)

    def _on_exit(self) -> None:
        self._hyper_parameters_widget.save_to_file(self._hyper_parameter_save_path)
