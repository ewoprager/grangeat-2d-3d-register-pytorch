from typing import Tuple, Callable, Any
import inspect
import logging

import torch
import matplotlib.pyplot as plt

from reg23_experiments.registration.lib.structs import SceneGeometry, Transformation, Sinogram2dGrid
from reg23_experiments.registration.lib.sinogram import Sinogram, SinogramClassic
from reg23_experiments.registration.lib import grangeat
from reg23_experiments.registration.lib import similarity_metric

__all__ = ["evaluate", "evaluate_direct"]

logger = logging.getLogger(__name__)


class ParametrisedSimilarityMetric:
    def __init__(self, underlying_function: Callable, **kwargs):
        # filter out key-word arguments that the function doesn't accept
        self._underlying_function = underlying_function
        sig = inspect.signature(self._underlying_function)
        self._kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    @property
    def func(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return lambda a, b: self._underlying_function(a, b, **self._kwargs)

    @property
    def func_weighted(self) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None:
        if self._underlying_function == similarity_metric.ncc:
            weighted_function = similarity_metric.weighted_ncc
        elif self._underlying_function == similarity_metric.local_ncc:
            weighted_function = similarity_metric.weighted_local_ncc
        else:
            return None
        return lambda a, b, w: weighted_function(a, b, w, **self._kwargs)


def evaluate(fixed_image: torch.Tensor, sinogram3d: Sinogram, *, transformation: Transformation,
             scene_geometry: SceneGeometry, fixed_image_grid: Sinogram2dGrid, save: bool = False,
             smooth: float | None = None, plot: Tuple[float, float] | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    device = sinogram3d.device
    source_position = scene_geometry.source_position(device=device)
    p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
    ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=device).to(dtype=torch.float32))

    if smooth is not None and isinstance(sinogram3d, SinogramClassic):
        resampled = sinogram3d.resample_python(ph_matrix=ph_matrix, fixed_image_grid=fixed_image_grid, smooth=smooth,
                                               plot=plot is not None)
    else:
        if smooth is not None:
            logger.warning("Cannot resample smooth as not given a SinogramClassic")
        resampled = sinogram3d.resample(ph_matrix, fixed_image_grid)

    if plot is not None:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(resampled.clone().cpu(), vmin=plot[0], vmax=plot[1])
        axes.axis('square')
        axes.set_title(
            "d/dr R3 [mu] resampled with sample smoothing" if smooth is not None else "d/dr R3 [mu] resampled")
        axes.set_xlabel("r")
        axes.set_ylabel("phi")
        plt.colorbar(mesh)
        if save:
            plt.savefig(
                "data/temp/d_dr_R3_mu_resampled_with_sample_smoothing.pgf" if smooth else "data/temp/d_dr_R3_mu_resampled.pgf")

    return ncc(fixed_image,
               resampled), resampled  # return reg23.normalised_cross_correlation(fixed_image, resampled),  # resampled


def evaluate_direct(fixed_image: torch.Tensor, volume_data: torch.Tensor, *, transformation: Transformation,
                    scene_geometry: SceneGeometry, fixed_image_grid: Sinogram2dGrid, voxel_spacing: torch.Tensor,
                    plot: bool = False) -> torch.Tensor:
    device = volume_data.device
    source_position = scene_geometry.source_position(device=device)
    p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
    ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=device)).to(dtype=torch.float32)

    direct = grangeat.directly_calculate_radon_slice(volume_data, ph_matrix=ph_matrix, output_grid=fixed_image_grid,
                                                     voxel_spacing=voxel_spacing)
    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(direct.cpu())
        axes.axis('square')
        axes.set_title("d/dr R3 [mu] calculated directly")
        axes.set_xlabel("r")
        axes.set_ylabel("phi")
        plt.colorbar(mesh)

    return ncc(fixed_image, direct)
