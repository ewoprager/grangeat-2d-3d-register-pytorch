import copy
import time
import math
import logging
from datetime import datetime
from typing import Callable, Any, NamedTuple
from abc import ABC, abstractmethod
from enum import Enum

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
import pathlib

from registration.lib.structs import Transformation, GrowingTensor
from registration.interface.transformations import TransformationWidget
from registration.interface.lib.structs import HyperParameters, WidgetSelectData, WidgetManageSaved, Cropping, \
    SavedXRayParams, Target
from registration.lib import optimisation
from registration.interface.registration_data import RegistrationData


def mapping_transformation_to_parameters(transformation: Transformation) -> torch.Tensor:
    ret = transformation.vectorised()
    ret[0:3] *= 100.0
    ret[5] /= 10.0
    return ret


def mapping_parameters_to_transformation(params: torch.Tensor) -> Transformation:
    params = params.clone()
    params[0:3] /= 100.0
    params[5] *= 10.0
    return Transformation.from_vector(params)


class OptimisationResult(NamedTuple):
    params: torch.Tensor
    param_history: torch.Tensor
    value_history: torch.Tensor


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
            iteration_callback: Callable[[torch.Tensor, torch.Tensor], None]) -> OptimisationResult:
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
            iteration_callback: Callable[[torch.Tensor, torch.Tensor], None]) -> OptimisationResult:
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
        initial_positions = np.random.normal(loc=starting_parameters.numpy(), size=(self.particle_count, n_dimensions),
                                             scale=1.0)
        initial_positions[0] = starting_parameters.numpy()
        optimiser = pyswarms.single.GlobalBestPSO(n_particles=self.particle_count, dimensions=n_dimensions,
                                                  init_pos=initial_positions, options=options)

        cost, converged_params = optimiser.optimize(objective_pso, iters=self.iteration_count)
        return OptimisationResult(params=torch.from_numpy(converged_params), param_history=param_history.get(),
                                  value_history=value_history.get())


class LocalSearch(OptimisationAlgorithm):
    def __init__(self, no_improvement_threshold: int, max_reductions: int):
        self._no_improvement_threshold = no_improvement_threshold
        self._max_reductions = max_reductions

    @property
    def no_improvement_threshold(self) -> int:
        return self._no_improvement_threshold

    @property
    def max_reductions(self) -> int:
        return self._max_reductions

    def __str__(self) -> str:
        return "Local search"

    @staticmethod
    def algorithm_name() -> str:
        return "Local Search"

    def run(self, *, starting_parameters: torch.Tensor, objective_function: Callable[[torch.Tensor], torch.Tensor],
            iteration_callback: Callable[[torch.Tensor, torch.Tensor], None]) -> OptimisationResult:
        n_dimensions = starting_parameters.numel()
        param_history = GrowingTensor([n_dimensions], 50)
        value_history = GrowingTensor([], 50)

        param_history.push_back(starting_parameters)
        value_history.push_back(objective_function(starting_parameters))

        def obj_func(params: torch.Tensor) -> torch.Tensor:
            params = torch.tensor(copy.deepcopy(params))
            param_history.push_back(params)
            value = objective_function(params)
            value_history.push_back(value)
            iteration_callback(param_history.get(), value_history.get())
            return value

        tic = time.time()
        res = optimisation.local_search(starting_position=starting_parameters,
                                        initial_step_size=torch.tensor([0.1, 0.1, 0.1, 2.0, 2.0, 2.0]),
                                        objective_function=obj_func,
                                        no_improvement_threshold=self.no_improvement_threshold,
                                        max_reductions=self.max_reductions)
        toc = time.time()
        logger.info("Done. Took {:.3f}s.".format(toc - tic))
        return OptimisationResult(res, param_history=param_history.get(), value_history=value_history.get())

        # def objective_scipy(params: np.ndarray) -> float:  #     params = torch.tensor(copy.deepcopy(params))  #  #  # param_history.push_back(params)  #     value = objective_function(params)  #     value_history.push_back(  # value)  #     iteration_callback(param_history.get(), value_history.get())  #     return value.item()  #  #  # tic = time.time()  # res = scipy.optimize.minimize(objective_scipy, starting_parameters, method='Powell')  # toc = time.time()  # logger.info("Done. Took {:.3f}s.".format(toc - tic))  # logger.info(res)  # return  #  # torch.from_numpy(res.x)


class Worker(QObject):
    finished = pyqtSignal(Transformation)
    progress = pyqtSignal(torch.Tensor, torch.Tensor, bool)

    def __init__(self, *, initial_transformation: Transformation,
                 objective_function: Callable[[Transformation], torch.Tensor], optimisation_algorithm: Any):
        super().__init__()
        self._initial_transformation = initial_transformation
        self._objective_function = objective_function
        self._optimisation_algorithm = optimisation_algorithm

    def run(self):
        def obj_func(params: torch.Tensor) -> torch.Tensor:
            return self._objective_function(mapping_parameters_to_transformation(params))

        starting_parameters: torch.Tensor = mapping_transformation_to_parameters(self._initial_transformation).cpu()

        logger.info("Optimising with '{}'...".format(str(self._optimisation_algorithm)))
        optimisation_result = self._optimisation_algorithm.run(starting_parameters=starting_parameters,
                                                               objective_function=obj_func,
                                                               iteration_callback=self._iteration_callback)
        self.progress.emit(optimisation_result.param_history, optimisation_result.value_history, True)
        logger.info("Optimisation finished.")
        res = mapping_parameters_to_transformation(optimisation_result.params)
        self.finished.emit(res)

    def _iteration_callback(self, param_history: torch.Tensor, value_history: torch.Tensor) -> None:
        self.progress.emit(param_history, value_history, False)


class OpAlgoWidget(ABC):
    @abstractmethod
    def get_op_algo(self) -> OptimisationAlgorithm:
        pass

    @abstractmethod
    def set_from_op_algo(self, _: Any) -> None:
        pass


class PSOWidget(widgets.Container, OpAlgoWidget):
    def __init__(self, *, particle_count: int, iteration_count: int):
        super().__init__(layout="horizontal")

        self._particle_count_widget = widgets.SpinBox(value=particle_count, min=1, max=5000, step=1,
                                                      label="N particles")
        self.append(self._particle_count_widget)

        self._iteration_count_widget = widgets.SpinBox(value=iteration_count, min=1, max=30, step=1,
                                                       label="N iterations")
        self.append(self._iteration_count_widget)

    def get_op_algo(self) -> ParticleSwarm:
        return ParticleSwarm(particle_count=self._particle_count_widget.get_value(),
                             iteration_count=self._iteration_count_widget.get_value())

    def set_from_op_algo(self, particle_swarm: ParticleSwarm) -> None:
        self._particle_count_widget.set_value(particle_swarm.particle_count)
        self._iteration_count_widget.set_value(particle_swarm.iteration_count)


class LocalSearchWidget(widgets.Container, OpAlgoWidget):
    def __init__(self, *, no_improvement_threshold: int, max_reductions: int):
        super().__init__(layout="horizontal")

        self._no_improvement_threshold_widget = widgets.SpinBox(value=no_improvement_threshold, min=2, max=1000, step=1,
                                                                label="reduction thresh.")
        self.append(self._no_improvement_threshold_widget)

        self._max_reductions_widget = widgets.SpinBox(value=max_reductions, min=0, max=500, step=1,
                                                      label="max reductions")
        self.append(self._max_reductions_widget)

    def get_op_algo(self) -> LocalSearch:
        return LocalSearch(no_improvement_threshold=self._no_improvement_threshold_widget.get_value(),
                           max_reductions=self._max_reductions_widget.get_value())

    def set_from_op_algo(self, local_search: LocalSearch) -> None:
        self._no_improvement_threshold_widget.set_value(local_search.no_improvement_threshold)
        self._max_reductions_widget.set_value(local_search.max_reductions)


class RegisterWidget(widgets.Container):
    def __init__(self, *, transformation_widget: TransformationWidget, registration_data: RegistrationData,
                 objective_functions: dict[str, Callable[[Transformation], torch.Tensor]],
                 save_directory: str | pathlib.Path, fixed_image_size: torch.Size):
        super().__init__(labels=False)
        self._transformation_widget = transformation_widget
        self._registration_data = registration_data
        self._objective_functions = objective_functions
        save_directory = pathlib.Path(save_directory)
        self._hyper_parameter_save_path = save_directory / "hyperparameter_library.pkl"
        self._xray_params_save_path = save_directory / "xray_params_library.pkl"

        # Optimisation worker thread and plotting
        self._evals_per_render: int = 2000
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
        self.append(widgets.Container(
            widgets=[self._bottom_crop_slider, self._top_crop_slider, self._right_crop_slider, self._left_crop_slider],
            layout="vertical"))

        ##
        ## Hyper-parameter saving and loading
        ##
        self._hyper_parameters_widget = WidgetManageSaved(initial_choices={"initial": self._current_hyper_parameters()},
                                                          DataType=HyperParameters,
                                                          load_from_file=self._hyper_parameter_save_path,
                                                          get_current_callback=self._current_hyper_parameters,
                                                          set_callback=self._set_hyper_parameters)
        self.append(self._hyper_parameters_widget)

        self._flip_target_button = widgets.Button(label="Flip")
        self._flip_target_button.changed.connect(self._on_flip_target_button)

        self._save_params_for_xray_button = widgets.PushButton(label="Save params for X-ray")
        self._save_params_for_xray_button.changed.connect(self._on_save_params_for_xray_button)

        self._load_params_for_xray_button = widgets.PushButton(label="Load params for X-ray")
        self._load_params_for_xray_button.changed.connect(self._on_load_params_for_xray_button)
        self.append(widgets.Container(
            widgets=[self._flip_target_button, self._save_params_for_xray_button, self._load_params_for_xray_button],
            layout="horizontal"))

        ##
        ## Objective function
        ##
        self._objective_function_widget = WidgetSelectData(widget_type=widgets.ComboBox,
                                                           initial_choices=objective_functions, label="Obj. func.")

        self._eval_once_button = widgets.PushButton(label="Evaluate once")
        self._eval_once_button.changed.connect(self._on_eval_once)

        self._eval_result_label = widgets.Label(label="Result:", value="n/a")

        self.append(widgets.Container(
            widgets=[self._objective_function_widget.widget, self._eval_once_button, self._eval_result_label],
            layout="horizontal", label="Obj. func."))

        ##
        ## Optimisation algorithm and parameters
        ##
        self._op_algo_widgets = {ParticleSwarm.algorithm_name(): PSOWidget(particle_count=2000, iteration_count=10),
                                 LocalSearch.algorithm_name(): LocalSearchWidget(no_improvement_threshold=10,
                                                                                 max_reductions=4)}
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

        self._evals_per_render_widget = widgets.SpinBox(value=self._evals_per_render, min=1, max=10000, step=1,
                                                        label="Evals./re-plot")
        self._evals_per_render_widget.changed.connect(self._on_evals_per_render)

        self.append(
            widgets.Container(widgets=[self._register_button, self._register_progress, self._evals_per_render_widget],
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
        self._worker = Worker(initial_transformation=self._transformation_widget.get_current_transformation(),
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

    def _iteration_callback(self, param_history: torch.Tensor, value_history: torch.Tensor, force_render: bool) -> None:
        self._iteration_callback_count += 1

        if not force_render and self._iteration_callback_count - self._last_rendered_iteration < self._evals_per_render:
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
        best_transformation = mapping_parameters_to_transformation(param_history[best_index])
        self._transformation_widget.set_current_transformation(best_transformation)

        self._register_progress.value = "{} evaluations, best = {:.4f}".format(self._iteration_callback_count,
                                                                               0.0 if self._best is None else self._best)

    def _finish_callback(self, converged_transformation: Transformation):
        self._transformation_widget.set_current_transformation(converged_transformation)
        self._transformation_widget.save_transformation(converged_transformation, "registration result {}".format(
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
        self._registration_data.hyperparameters = self._current_hyper_parameters()
        self._registration_data.refresh_hyperparameter_dependent()
        # self._fixed_image_crop_callback(
        #     Cropping(top=self._top_crop_slider.get_value(), bottom=self._bottom_crop_slider.get_value(),
        #              left=self._left_crop_slider.get_value(), right=self._right_crop_slider.get_value()))
        self._ignore_crop_sliders = False

    def _current_hyper_parameters(self) -> HyperParameters:
        return HyperParameters(
            cropping=Cropping(right=self._right_crop_slider.get_value(), top=self._top_crop_slider.get_value(),
                              left=self._left_crop_slider.get_value(), bottom=self._bottom_crop_slider.get_value()),
            source_offset=torch.zeros(2))

    def _set_hyper_parameters(self, new_value: HyperParameters) -> None:
        if new_value.cropping.right > self._right_crop_slider.max:
            self._right_crop_slider.max = new_value.cropping.right
        if new_value.cropping.right < self._right_crop_slider.min:
            self._right_crop_slider.min = new_value.cropping.right
        self._right_crop_slider.set_value(new_value.cropping.right)

        if new_value.cropping.top > self._top_crop_slider.max:
            self._top_crop_slider.max = new_value.cropping.top
        if new_value.cropping.top < self._top_crop_slider.min:
            self._top_crop_slider.min = new_value.cropping.top
        self._top_crop_slider.set_value(new_value.cropping.top)

        if new_value.cropping.left > self._left_crop_slider.max:
            self._left_crop_slider.max = new_value.cropping.left
        if new_value.cropping.left < self._left_crop_slider.min:
            self._left_crop_slider.min = new_value.cropping.left
        self._left_crop_slider.set_value(new_value.cropping.left)

        if new_value.cropping.bottom > self._bottom_crop_slider.max:
            self._bottom_crop_slider.max = new_value.cropping.bottom
        if new_value.cropping.bottom < self._bottom_crop_slider.min:
            self._bottom_crop_slider.min = new_value.cropping.bottom
        self._bottom_crop_slider.set_value(new_value.cropping.bottom)

    def _on_save_params_for_xray_button(self) -> None:
        if self._registration_data.target.xray_path is None:
            logger.warning("No X-ray path defined, cannot save associated parameters.")
            return
        current_saved = dict({})
        if self._xray_params_save_path.is_file():
            current_saved = torch.load(self._xray_params_save_path, weights_only=False)
            if not isinstance(current_saved, dict):
                new_name = self._xray_params_save_path.with_stem(
                    "invalid_type_{}".format(self._xray_params_save_path.stem))
                logger.warning("Existing X-ray param library '{}' is of invalid type '{}'; renaming to '{}'"
                               "".format(str(self._xray_params_save_path), type(current_saved).__name__, str(new_name)))
                self._xray_params_save_path.rename(new_name)
                current_saved = dict({})
        to_save = SavedXRayParams(transformation=self._transformation_widget.get_current_transformation(),
                                  hyperparameters=self._current_hyper_parameters(),
                                  flipped=self._registration_data.target.flipped)
        current_saved[self._registration_data.target.xray_path] = to_save
        torch.save(current_saved, self._xray_params_save_path)
        logger.info("Saved parameters:\n{}\nfor X-ray '{}'."
                    "".format(to_save, str(self._registration_data.target.xray_path)))

    def _on_load_params_for_xray_button(self) -> None:
        if self._registration_data.target.xray_path is None:
            logger.warning("No X-ray path defined, cannot load associated parameters.")
            return
        if not self._xray_params_save_path.is_file():
            logger.warning("No X-ray parameter save file '{}' found; cannot load parameters."
                           "".format(str(self._xray_params_save_path)))
            return
        current_saved = torch.load(self._xray_params_save_path, weights_only=False)
        if not isinstance(current_saved, dict):
            logger.warning("X-ray parameter save file '{}' has invalid type '{}'; cannot load parameters."
                           "".format(str(self._xray_params_save_path), type(current_saved).__name__))
            return
        if self._registration_data.target.xray_path not in current_saved:
            logger.warning("No parameters saved for current X-ray in parameter save file '{}'; cannot load parameters."
                           "".format(str(self._xray_params_save_path)))
            return
        loaded = current_saved[self._registration_data.target.xray_path]
        self._set_hyper_parameters(loaded.hyperparameters)
        self._transformation_widget.set_current_transformation(loaded.transformation)
        self._registration_data.target = Target(xray_path=self._registration_data.target.xray_path,
                                                flipped=loaded.flipped)

    def _on_flip_target_button(self) -> None:
        logger.info("Setting target flip to {}".format(str(not self._registration_data.target.flipped)))
        self._registration_data.target = Target(xray_path=self._registration_data.target.xray_path,
                                                flipped=not self._registration_data.target.flipped)
        self._registration_data.refresh_target_dependent()

    def _on_exit(self) -> None:
        self._hyper_parameters_widget.save_to_file(self._hyper_parameter_save_path)
