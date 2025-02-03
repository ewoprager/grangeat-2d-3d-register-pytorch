from typing import Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as pgo
import time
import torch
import nrrd

import Extension as ExtensionTest

TaskSummaryRadon3D = Tuple[str, torch.Tensor]


def read_nrrd(path: str, downsample_factor=1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print("Loading CT data file {}...".format(path))
    data, header = nrrd.read(path)
    print("Done.")
    print("Processing CT data...")
    sizes = header['sizes']
    print("\tVolume size = [{} x {} x {}]".format(sizes[0], sizes[1], sizes[2]))
    data = torch.tensor(data, device="cpu")
    image = torch.maximum(data.type(torch.float32) + 1000., torch.tensor([0.], device=data.device))
    if downsample_factor > 1:
        down_sampler = torch.nn.AvgPool3d(downsample_factor)
        image = down_sampler(image[None, :, :, :])[0]
    sizes = image.size()
    print("\tVolume size after down-sampling = [{} x {} x {}]".format(sizes[0], sizes[1], sizes[2]))
    bounds = torch.Tensor([image.min().item(), image.max().item()])
    print("\tValue range = ({:.3f}, {:.3f})".format(bounds[0], bounds[1]))
    bounds[1] *= 10000.
    directions = torch.tensor(header['space directions'])
    spacing = float(downsample_factor) * directions.norm(dim=1)
    print("\tCT voxel spacing = [{} x {} x {}] mm".format(spacing[0], spacing[1], spacing[2]))
    print("Done.")

    return image, spacing, bounds


def task_radon3d(function, name: str, device: str, image: torch.Tensor, spacing: torch.Tensor,
                 output_size: torch.Tensor) -> TaskSummaryRadon3D:
    image_devices = image.to(device=device)
    phi_count = output_size[0].item()
    theta_count = output_size[1].item()
    r_count = output_size[2].item()
    phi_values = torch.linspace(-.5 * torch.pi, .5 * torch.pi, phi_count, device=device)
    theta_values = torch.linspace(-.5 * torch.pi, .5 * torch.pi, theta_count, device=device)
    image_depth: torch.Tensor = spacing[0] * float(image.size()[0])
    image_height: torch.Tensor = spacing[1] * float(image.size()[1])
    image_width: torch.Tensor = spacing[2] * float(image.size()[2])
    image_diag = torch.sqrt(image_depth.square() + image_height.square() + image_width.square()).item()
    r_values = torch.linspace(-.5 * image_diag, .5 * image_diag, r_count, device=device)
    output = function(image_devices, spacing[0], spacing[1], spacing[2], phi_values, theta_values, r_values, 64)
    name: str = "{}_on_{}".format(name, device)
    return name, output.cpu()


def plot_task_radon3d(summary: TaskSummaryRadon3D, bounds: torch.Tensor):
    size = summary[1].size()
    X, Y, Z = torch.meshgrid([torch.arange(0, size[0], 1), torch.arange(0, size[1], 1), torch.arange(0, size[2], 1)])
    fig = pgo.Figure(data=pgo.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=summary[1].flatten(),
                                     isomin=bounds[0].item(), isomax=bounds[1].item(), opacity=.2, surface_count=21),
                     layout=pgo.Layout(title=summary[0]))
    fig.show()


def run_task(task, task_plot, function, name: str, device: str, image: torch.Tensor, spacing: torch.Tensor,
             output_size: torch.Tensor, bounds: torch.Tensor):
    print("Running {} on {}...".format(name, device))
    tic = time.time()
    summary = task(function, name, device, image, spacing, output_size)
    toc = time.time()
    print("Done; took {:.3f}s. Saving and plotting...".format(toc - tic))
    torch.save(summary[1], "cache/{}.pt".format(summary[0]))
    task_plot(summary, bounds)
    print("Done.")
    return summary


def benchmark_radon3d(path: str):
    print("----- Benchmarking radon3d -----")

    image = torch.zeros((5, 5, 5))
    image[0, 0, 0] = 1.
    image[4, 3, 2] = .5

    spacing = torch.tensor([1., 1., 1.])
    bounds = torch.tensor([image.min(), 5. * image.max()])

    # image, spacing, bounds = read_nrrd(path, 8)

    output_size = torch.tensor([64, 64, 64])

    outputs: list[TaskSummaryRadon3D] = [
        run_task(task_radon3d, plot_task_radon3d, ExtensionTest.radon3d, "RT3 V1", "cpu", image, spacing, output_size,
                 bounds),
        run_task(task_radon3d, plot_task_radon3d, ExtensionTest.radon3d, "RT3 V1", "cuda", image, spacing, output_size,
                 bounds),
        run_task(task_radon3d, plot_task_radon3d, ExtensionTest.radon3d_v2, "RT3 V2", "cuda", image, spacing,
                 output_size, bounds)]

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


def benchmark_dRadon3dDR(path: str):
    print("----- Benchmarking dRadon3dDR -----")

    # image, spacing, bounds = read_nrrd(path, 1)

    image = torch.zeros((5, 5, 5))
    image[0, 0, 0] = 1.
    image[4, 3, 2] = .5

    spacing = torch.tensor([1., 1., 1.])
    bounds = torch.tensor([-2. * image.max(), 2. * image.max()])

    output_size = torch.tensor([100, 100, 100])

    outputs: list[TaskSummaryRadon3D] = [
        run_task(task_radon3d, plot_task_radon3d, ExtensionTest.dRadon3dDR_v2, "dRT3-dR V2", "cuda", image, spacing,
                 output_size, bounds),
        run_task(task_radon3d, plot_task_radon3d, ExtensionTest.dRadon3dDR, "dRT3-dR V1", "cuda", image, spacing,
                 output_size, bounds)]

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
