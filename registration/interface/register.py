import copy
import time
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

import torch
import napari
from magicgui import magicgui, widgets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQT
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from registration.lib.structs import *
from registration.lib.geometry import generate_drr
from registration.objective_function import zncc
from registration.interface.transformations import TransformationManager


def objective_function_standard(*, fixed_image: torch.Tensor, volume: torch.Tensor, voxel_spacing: torch.Tensor,
                                detector_spacing: torch.Tensor, transformation: Transformation,
                                scene_geometry: SceneGeometry) -> torch.Tensor:
    assert len(fixed_image.size()) == 2
    assert len(volume.size()) == 3
    assert voxel_spacing.size() == torch.Size([3])
    assert detector_spacing.size() == torch.Size([2])
    device = volume.device
    assert fixed_image.device == device
    assert transformation.device_consistent()
    assert transformation.device() == device

    moving_image = generate_drr(volume, transformation=transformation, voxel_spacing=voxel_spacing,
                                detector_spacing=detector_spacing, scene_geometry=scene_geometry,
                                output_size=fixed_image.size())

    return zncc(fixed_image, moving_image)


def register(fixed_image: torch.Tensor, volume: torch.Tensor, voxel_spacing: torch.Tensor,
             detector_spacing: torch.Tensor, initial_transformation: Transformation, scene_geometry: SceneGeometry,
             it_callback: Callable[[list, list, Transformation], None]) -> Transformation:
    device = volume.device

    def objective(params: torch.Tensor) -> torch.Tensor:
        return -objective_function_standard(fixed_image=fixed_image, volume=volume, voxel_spacing=voxel_spacing,
                                            detector_spacing=detector_spacing,
                                            transformation=Transformation(params[0:3], params[3:6]).to(device=device),
                                            scene_geometry=scene_geometry)

    logger.info("Optimising...")
    param_history = []
    value_history = []
    start_params: torch.Tensor = initial_transformation.vectorised()

    def objective_scipy(params: np.ndarray) -> float:
        params = torch.tensor(copy.deepcopy(params))
        param_history.append(params)
        value = objective(params)
        value_history.append(value)
        it_callback(param_history, value_history, Transformation(params[0:3], params[3:6]).to(device=device))
        return value.item()

    tic = time.time()
    # res = scipy.optimize.minimize(objective_scipy, start_params.numpy(), method='Powell')
    res = scipy.optimize.basinhopping(objective_scipy, start_params.cpu().numpy(), T=5.0,
                                      minimizer_kwargs={"method": 'Nelder-Mead'})
    toc = time.time()
    logger.info("Done. Took {:.3f}s.".format(toc - tic))
    logger.info(res)
    converged_params = torch.from_numpy(res.x)
    return Transformation(converged_params[0:3], converged_params[3:6]).to(device=device)


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(list, list, Transformation)

    def __init__(self, fixed_image: torch.Tensor, volume: torch.Tensor, voxel_spacing: torch.Tensor,
                 detector_spacing: torch.Tensor, initial_transformation: Transformation, scene_geometry: SceneGeometry):
        super().__init__()
        self.fixed_image = fixed_image
        self.volume = volume
        self.voxel_spacing = voxel_spacing
        self.detector_spacing = detector_spacing
        self.scene_geometry = scene_geometry
        self.initial_transformation = initial_transformation


    def run(self):
        def iteration_callback(param_history: list, value_history: list, latest_transformation: Transformation) -> None:
            nonlocal self
            self.progress.emit(param_history, value_history, latest_transformation)

        register(self.fixed_image, self.volume, self.voxel_spacing, self.detector_spacing, self.initial_transformation,
                 self.scene_geometry, iteration_callback)
        self.finished.emit()


class RegisterWidget(widgets.Container):
    def __init__(self, fixed_image: torch.Tensor, volume: torch.Tensor, voxel_spacing: torch.Tensor,
                 detector_spacing: torch.Tensor, transformation_manager: TransformationManager,
                 scene_geometry: SceneGeometry):
        super().__init__()
        self.fixed_image = fixed_image
        self.volume = volume
        self.voxel_spacing = voxel_spacing
        self.detector_spacing = detector_spacing
        self.scene_geometry = scene_geometry
        self.transformation_manager = transformation_manager

        self.fig, self.axes = plt.subplots()
        register_button = widgets.PushButton(label="Register")

        def iteration_callback(param_history: list, value_history: list, latest_transformation: Transformation) -> None:
            nonlocal self
            # self.axes.cla()
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

            self.transformation_manager.set_transformation(latest_transformation)

        @register_button.changed.connect
        def _():
            nonlocal self
            thread = QThread()
            worker = Worker(self.fixed_image, self.volume, self.voxel_spacing, self.detector_spacing,
                            self.transformation_manager.get_current_transformation(), self.scene_geometry)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            worker.progress.connect(iteration_callback)
            thread.start()

        self.native.layout().addWidget(FigureCanvasQT(self.fig))
        self.append(register_button)


def build_register_widget(fixed_image: torch.Tensor, volume: torch.Tensor, voxel_spacing: torch.Tensor,
                          detector_spacing: torch.Tensor, transformation_manager: TransformationManager,
                          scene_geometry: SceneGeometry) -> widgets.Widget:
    return RegisterWidget(fixed_image, volume, voxel_spacing, detector_spacing, transformation_manager, scene_geometry)
