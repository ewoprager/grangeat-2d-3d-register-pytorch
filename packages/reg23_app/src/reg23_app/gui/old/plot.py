import copy
import logging
from typing import Callable, NamedTuple, Tuple
from enum import Enum
import pathlib

import torch
from magicgui import widgets
import matplotlib.pyplot as plt
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from tqdm import tqdm
from matplotlib import cm

from reg23_experiments.data.structs import Transformation
from reg23_experiments.ui.old.transformations import TransformationWidget
from reg23_experiments.ui.old.lib.structs import WidgetSelectData

__all__ = ["TransformationParameterType", "TransformationParameter", "ParameterRange", "landscape2", "WorkSpec",
           "Landscape2", "four_landscapes", "Worker", "PlotWidget"]

logger = logging.getLogger(__name__)


class TransformationParameterType(Enum):
    ROTATION = 0
    TRANSLATION = 1


class TransformationParameter(NamedTuple):
    type: TransformationParameterType
    index: int

    def get(self, transformation: Transformation) -> torch.Tensor:
        if self.type == TransformationParameterType.ROTATION:
            return transformation.rotation[self.index]
        else:
            assert self.type == TransformationParameterType.TRANSLATION
            return transformation.translation[self.index]

    def set(self, transformation: Transformation, value: torch.Tensor) -> None:
        if self.type == TransformationParameterType.ROTATION:
            transformation.rotation[self.index] = value
        else:
            assert self.type == TransformationParameterType.TRANSLATION
            transformation.translation[self.index] = value

    def __str__(self) -> str:
        return "{} {}".format(["X", "Y", "Z"][self.index], ["rotation", "translation"][self.type.value])


class ParameterRange(NamedTuple):
    parameter: TransformationParameter
    range: torch.Tensor
    count: int

    def get_grid_around_centre(self, transformation_centre: Transformation) -> torch.Tensor:
        centre = self.parameter.get(transformation_centre)
        return torch.linspace(centre - 0.5 * self.range, centre + 0.5 * self.range, self.count)


def landscape2(*, objective_function: Callable[[Transformation], torch.Tensor], central_transformation: Transformation,
               x_range: ParameterRange, y_range: ParameterRange) -> torch.Tensor:
    output_size = torch.Size([y_range.count, x_range.count])
    output = torch.zeros(output_size)
    y_grid = y_range.get_grid_around_centre(central_transformation)
    x_grid = x_range.get_grid_around_centre(central_transformation)
    logger.info("Evaluating landscape...")
    for j in tqdm(range(y_range.count), desc="Progress through rows of 2D grid"):
        for i in range(x_range.count):
            transformation = copy.deepcopy(central_transformation)
            x_range.parameter.set(transformation, x_grid[i])
            y_range.parameter.set(transformation, y_grid[j])
            output[j, i] = objective_function(transformation)
    logger.info("Landscape evaluated.")
    return output


class WorkSpec(NamedTuple):
    objective_function: Callable[[Transformation], torch.Tensor]
    central_transformation: Transformation
    transformation_range: Transformation
    counts: int


class Landscape2(NamedTuple):
    x_range: ParameterRange
    y_range: ParameterRange
    data: torch.Tensor


def four_landscapes(work_spec: WorkSpec) -> Tuple[Transformation, list[Landscape2]]:
    ret = []
    x_param = TransformationParameter(type=TransformationParameterType.ROTATION, index=0)
    y_param = TransformationParameter(type=TransformationParameterType.ROTATION, index=1)
    x_range = ParameterRange(parameter=x_param, range=x_param.get(work_spec.transformation_range),
                             count=work_spec.counts)
    y_range = ParameterRange(parameter=y_param, range=y_param.get(work_spec.transformation_range),
                             count=work_spec.counts)
    ret.append(Landscape2(x_range, y_range, landscape2(objective_function=work_spec.objective_function,
                                                       central_transformation=work_spec.central_transformation,
                                                       x_range=x_range, y_range=y_range)))

    x_param = TransformationParameter(type=TransformationParameterType.ROTATION, index=1)
    y_param = TransformationParameter(type=TransformationParameterType.ROTATION, index=2)
    x_range = ParameterRange(parameter=x_param, range=x_param.get(work_spec.transformation_range),
                             count=work_spec.counts)
    y_range = ParameterRange(parameter=y_param, range=y_param.get(work_spec.transformation_range),
                             count=work_spec.counts)
    ret.append(Landscape2(x_range, y_range, landscape2(objective_function=work_spec.objective_function,
                                                       central_transformation=work_spec.central_transformation,
                                                       x_range=x_range, y_range=y_range)))

    x_param = TransformationParameter(type=TransformationParameterType.TRANSLATION, index=0)
    y_param = TransformationParameter(type=TransformationParameterType.TRANSLATION, index=1)
    x_range = ParameterRange(parameter=x_param, range=x_param.get(work_spec.transformation_range),
                             count=work_spec.counts)
    y_range = ParameterRange(parameter=y_param, range=y_param.get(work_spec.transformation_range),
                             count=work_spec.counts)
    ret.append(Landscape2(x_range, y_range, landscape2(objective_function=work_spec.objective_function,
                                                       central_transformation=work_spec.central_transformation,
                                                       x_range=x_range, y_range=y_range)))

    x_param = TransformationParameter(type=TransformationParameterType.TRANSLATION, index=1)
    y_param = TransformationParameter(type=TransformationParameterType.TRANSLATION, index=2)
    x_range = ParameterRange(parameter=x_param, range=x_param.get(work_spec.transformation_range),
                             count=work_spec.counts)
    y_range = ParameterRange(parameter=y_param, range=y_param.get(work_spec.transformation_range),
                             count=work_spec.counts)
    ret.append(Landscape2(x_range, y_range, landscape2(objective_function=work_spec.objective_function,
                                                       central_transformation=work_spec.central_transformation,
                                                       x_range=x_range, y_range=y_range)))

    return work_spec.central_transformation, ret


class Worker(QObject):
    finished = pyqtSignal(Transformation, list)

    def __init__(self, work_spec: WorkSpec):
        super().__init__()
        self._work_spec = work_spec

    def run(self):
        res = four_landscapes(self._work_spec)
        self.finished.emit(*res)


class PlotWidget(widgets.Container):
    def __init__(self, *, transformation_widget: TransformationWidget,
                 objective_functions: dict[str, Callable[[Transformation], torch.Tensor]], window):
        super().__init__()
        self._transformation_widget = transformation_widget
        self._objective_functions = objective_functions
        self._window = window

        self._objective_function_widget = WidgetSelectData(widget_type=widgets.ComboBox,
                                                           initial_choices=objective_functions, label="Obj. func.")
        self.append(self._objective_function_widget.widget)

        self._translation_range_widgets = [widgets.SpinBox(value=30, min=1, max=100, step=1, label=s) for s in
                                           ["X", "Y", "Z"]]
        self._rotation_range_widgets = [widgets.FloatSpinBox(value=1.4, min=0.2, max=3.2, step=0.2, label=s) for s in
                                        ["X", "Y", "Z"]]
        self.append(widgets.Container(
            widgets=[widgets.Label(label="Translation [mm]")] + self._translation_range_widgets + [
                widgets.Label(label="Rotation [rad]")] + self._rotation_range_widgets, label="Ranges"))

        self._counts_widget = widgets.SpinBox(value=30, min=2, max=200, step=1, label="Eval. counts")
        self.append(self._counts_widget)

        self._evaluate_button = widgets.PushButton(label="Evaluate")
        self._evaluate_button.changed.connect(self._on_evaluate)
        self.append(self._evaluate_button)

    def _current_obj_func(self) -> Callable[[Transformation], torch.Tensor] | None:
        current = self._objective_function_widget.get_selected()  # from a ComboBox, a str is returned
        return self._objective_function_widget.get_data(current)

    def _current_transformation_range(self) -> Transformation:
        ret = Transformation.zero()
        for i in range(3):
            ret.translation[i] = self._translation_range_widgets[i].get_value()
        for i in range(3):
            ret.rotation[i] = self._rotation_range_widgets[i].get_value()
        return ret

    def _on_evaluate(self) -> None:
        self._thread = QThread()
        self._worker = Worker(WorkSpec(objective_function=self._current_obj_func(),
                                       central_transformation=self._transformation_widget.get_current_transformation(),
                                       transformation_range=self._current_transformation_range(),
                                       counts=self._counts_widget.get_value()))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._finish_callback)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _finish_callback(self, central_transformation: Transformation, landscapes: list[Landscape2]) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        figures_widget = widgets.Container(layout="horizontal", labels=False)
        for landscape in landscapes:
            fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
            figures_widget.native.layout().addWidget(FigureCanvasQTAgg(fig))
            xs = landscape.x_range.get_grid_around_centre(central_transformation)
            ys = landscape.y_range.get_grid_around_centre(central_transformation)
            ys, xs = torch.meshgrid(ys, xs)
            axes.cla()
            axes.plot_surface(xs.clone().detach().cpu().numpy(), ys.clone().detach().cpu().numpy(),
                              landscape.data.clone().detach().cpu().numpy(), cmap=cm.get_cmap("viridis"))
            axes.set_xlabel(str(landscape.x_range.parameter))
            axes.set_ylabel(str(landscape.y_range.parameter))
            axes.set_zlabel("objective function value")
            fname = "of_over_{}_{}.png".format(str(landscape.x_range.parameter).replace(' ', '_'),
                                               str(landscape.y_range.parameter).replace(' ', '_'))
            fig.savefig(pathlib.Path("figures") / "landscapes" / fname, dpi=300, bbox_inches='tight')
            fig.canvas.draw()

        self._window.add_dock_widget(widgets.Container(
            widgets=[widgets.Label(value="Landscape over two parameters around {}".format(str(central_transformation))),
                     figures_widget], labels=False), name="4 landscapes over two parameters each", area="right",
            tabify=True)
        logger.info("Landscape plotting finished")
