from typing import Tuple, NamedTuple
import matplotlib.pyplot as plt
import time
import torch

import Extension as ExtensionTest

from registration.lib.structs import *
import registration.data as data
import registration.pre_computed as pre_computed
from registration.lib.structs import LinearRange, SceneGeometry

TaskSummaryResample = Tuple[str, torch.Tensor]


class FunctionParams(NamedTuple):
    sinogram3d: torch.Tensor
    spacing: torch.Tensor
    range_centres: torch.Tensor
    projection_matrix: torch.Tensor
    phi_values: torch.Tensor
    r_values: torch.Tensor

    def to(self, **kwargs) -> 'FunctionParams':
        return FunctionParams(self.sinogram3d.to(**kwargs), self.spacing.to(**kwargs), self.range_centres.to(**kwargs),
                              self.projection_matrix.to(**kwargs), self.phi_values.to(**kwargs),
                              self.r_values.to(**kwargs))


def task_resample_sinogram3d(function, params: FunctionParams) -> TaskSummaryResample:
    output = function(params.sinogram3d, params.spacing, params.range_centres, params.projection_matrix,
                      params.phi_values, params.r_values)
    return output


def plot_task_resample_sinogram3d(summary: TaskSummaryResample):
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(summary[1].clone())
    axes.axis('square')
    axes.set_title("d/dr R3 [mu] resampled: {}".format(summary[0]))
    axes.set_xlabel("r")
    axes.set_ylabel("phi")
    plt.colorbar(mesh)


def run_task(task, task_plot, function, name: str, device: str, params: FunctionParams) -> TaskSummaryResample:
    params_device = params.to(device=device)
    print("Running {} on {}...".format(name, device))
    tic = time.time()
    output = task(function, params_device)
    toc = time.time()
    print("Done; took {:.3f}s. Saving and plotting...".format(toc - tic))
    name: str = "{}_on_{}".format(name, device)
    summary = name, output.cpu()
    torch.save(summary[1], "cache/{}.pt".format(summary[0]))
    task_plot(summary)
    print("Done.")
    return summary


def benchmark_resample_sinogram3d(path: str, *, cache_directory: str, load_cached: bool = True,
                                  save_to_cache: bool = True):
    print("----- Benchmarking resample_sinogram3d -----")

    cuda = torch.device('cuda')

    volume_spec = None
    sinogram3d = None
    sinogram3d_range = None
    if load_cached:
        volume_spec = data.load_cached_volume(cache_directory)

    if volume_spec is None:
        volume_downsample_factor: int = 4
    else:
        path, volume_downsample_factor, sinogram3d, sinogram3d_range = volume_spec

    if path is None:
        save_to_cache = False
        vol_data = torch.zeros((3, 3, 3))
        vol_data[1, 1, 1] = 1.
        voxel_spacing = torch.tensor([10., 10., 10.])
    else:
        vol_data, voxel_spacing, bounds = data.read_nrrd(path, downsample_factor=volume_downsample_factor)
        vol_data = vol_data.to(dtype=torch.float32)

    if sinogram3d is None or sinogram3d_range is None:
        sinogram3d, sinogram3d_range = pre_computed.calculate_volume_sinogram(cache_directory, vol_data.to(device=cuda),
                                                                              voxel_spacing.to(device=cuda), path,
                                                                              volume_downsample_factor,
                                                                              device=torch.device('cuda'),
                                                                              save_to_cache=save_to_cache,
                                                                              vol_counts=192)

    vol_diag: float = (
            torch.tensor([vol_data.size()], dtype=torch.float32) * voxel_spacing).square().sum().sqrt().item()

    phi_values = LinearRange(-.5 * torch.pi, .5 * torch.pi).generate_range(1000)
    r_values = LinearRange(-.5 * vol_diag, .5 * vol_diag).generate_range(1000)
    phi_values, r_values = torch.meshgrid(phi_values, r_values)

    scene_geometry = SceneGeometry(source_distance=1000.)
    transformation = Transformation(rotation=torch.tensor([0., 0., 0.]), translation=torch.tensor([0., 0., 0.]))
    source_position = scene_geometry.source_position()
    p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
    ph_matrix = torch.matmul(p_matrix, transformation.get_h()).to(dtype=torch.float32)

    params = FunctionParams(sinogram3d, sinogram3d_range.get_spacing(sinogram3d.size()), sinogram3d_range.get_centres(),
                            ph_matrix, phi_values, r_values)

    outputs: list[TaskSummaryResample] = [
        run_task(task_resample_sinogram3d, plot_task_resample_sinogram3d, ExtensionTest.resample_sinogram3d,
                 "ResampleSinogram3D", "cpu", params),
        run_task(task_resample_sinogram3d, plot_task_resample_sinogram3d, ExtensionTest.resample_sinogram3d,
                 "ResampleSinogram3D", "cuda", params)]

    plt.show()

    print("Calculating discrepancies...")
    found: bool = False
    for i in range(len(outputs) - 1):
        discrepancy = ((outputs[i][1] - outputs[i + 1][1]).abs() / (
                .5 * (outputs[i][1] + outputs[i + 1][1]).abs() + 1e-5)).mean()
        if discrepancy > 1e-2:
            found = True
            print("\tAverage discrepancy between outputs {} and {} is {:.3f} %".format(outputs[i][0], outputs[i + 1][0],
                                                                                       100. * discrepancy))
    if not found:
        print("\tNo discrepancies found.")
    print("Done.")

    # print("Showing plots...")  # X, Y, Z = torch.meshgrid([torch.arange(0, size[0], 1), torch.arange(0, size[1], 1), torch.arange(0, size[2], 1)])  # fig = pgo.Figure(  #     data=pgo.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=image.flatten(), isomin=.0, isomax=2000.,  #                     opacity=.1, surface_count=21), layout=pgo.Layout(title="Input"))  # fig.show()
