from typing import Tuple

import matplotlib.cm
import matplotlib.pyplot as plt
import time
import torch
import pydicom

import Extension as ExtensionTest

TaskSummaryRadon2D = Tuple[str, torch.Tensor]


def read_dicom(path: str, downsample_factor) -> Tuple[torch.Tensor, torch.Tensor]:
    print("Loading X-ray data file {}...".format(path))
    x_ray = pydicom.dcmread(path)
    print("Done.")
    print("Processing X-ray data...")
    data = torch.tensor(x_ray.pixel_array, device="cpu")
    sizes = data.size()
    print("\tX-ray size = [{} x {}]".format(sizes[0], sizes[1]))
    image = torch.maximum(data.type(torch.float32), torch.tensor([0.], device=data.device))
    if downsample_factor > 1:
        down_sampler = torch.nn.AvgPool2d(downsample_factor)
        image = down_sampler(image[None, :, :])[0]
    sizes = image.size()
    print("\tX-ray size after down-sampling = [{} x {}]".format(sizes[0], sizes[1]))
    bounds = torch.tensor([image.min().item(), image.max().item()])
    print("\tValue range = ({:.3f}, {:.3f})".format(bounds[0], bounds[1]))
    # bounds[1] *= 10000.
    directions = torch.tensor(x_ray.PixelSpacing)
    spacing = float(downsample_factor) * directions
    print("\tX-ray pixel spacing = [{} x {}] mm".format(spacing[0], spacing[1]))
    print("Done.")
    return image, spacing


def task_radon2d(function, name: str, device: str, image: torch.Tensor, spacing: torch.Tensor) -> TaskSummaryRadon2D:
    image_devices = image.to(device=device)
    phi_count = 1000
    r_count = 1000
    phi_values = torch.linspace(-.5 * torch.pi, .5 * torch.pi, phi_count, device=device)
    image_height: torch.Tensor = spacing[0] * float(image.size()[0])
    image_width: torch.Tensor = spacing[1] * float(image.size()[1])
    image_diag = torch.sqrt(image_height.square() + image_width.square()).item()
    r_values = torch.linspace(-.5 * image_diag, .5 * image_diag, r_count, device=device)
    phi_values, r_values = torch.meshgrid(phi_values, r_values)
    output = function(image_devices, spacing[0], spacing[1], phi_values, r_values, 1024)
    return "{} on {}".format(name, device), output.cpu()


def plot_task_radon2d(summary: TaskSummaryRadon2D):
    # hist = torch.histogram(summary[1].flatten(), 100)
    # plt.hist(hist.hist, hist.bin_edges)
    # plt.show()
    fig, axes = plt.subplots()
    mesh = axes.pcolormesh(summary[1])
    fig.colorbar(mesh)
    axes.set_title(summary[0])


def run_task(task, task_plot, function, name: str, device: str, image: torch.Tensor, spacing: torch.Tensor):
    print("Running {} on {}...".format(name, device))
    tic = time.time()
    summary = task(function, name, device, image, spacing)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    task_plot(summary)
    print("Done.")
    return summary


def benchmark_radon2d(path: str):
    print("----- Benchmarking radon2d -----")

    # image = torch.tensor(
    #     [[0.1, 0., 0.2, 0.4, 0.3, 0., .8], [0., 2., 0., 0.2, 0., 0., .8], [1., 0., 0.1, 0., 1., 0.4, .8],
    #      [0.1, 0.6, 0.4, 0.1, 0., 0.5, .8]])

    image, spacing = read_dicom(path, 1)

    outputs: list[TaskSummaryRadon2D] = [
        run_task(task_radon2d, plot_task_radon2d, ExtensionTest.radon2d, "RT2 V1", "cpu", image, spacing),
        run_task(task_radon2d, plot_task_radon2d, ExtensionTest.radon2d, "RT2 V1", "cuda", image, spacing),
        run_task(task_radon2d, plot_task_radon2d, ExtensionTest.radon2d_v2, "RT2 V2", "cpu", image, spacing),
        run_task(task_radon2d, plot_task_radon2d, ExtensionTest.radon2d_v2, "RT2 V2", "cuda", image, spacing)]

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

    print("Showing plots...")
    fig, axes = plt.subplots()
    mesh = axes.pcolormesh(image)
    fig.colorbar(mesh)
    axes.set_title("Input image")
    plt.show()


def benchmark_dRadon2dDR(path: str):
    print("----- Benchmarking dRadon2dDR -----")

    image, spacing = read_dicom(path, 1)

    outputs: list[TaskSummaryRadon2D] = [
        run_task(task_radon2d, plot_task_radon2d, ExtensionTest.dRadon2dDR, "dRT2-dR V1", "cpu", image, spacing),
        run_task(task_radon2d, plot_task_radon2d, ExtensionTest.dRadon2dDR, "dRT2-dR V1", "cuda", image, spacing)]

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

    print("Showing plots...")
    fig, axes = plt.subplots()
    mesh = axes.pcolormesh(image)
    fig.colorbar(mesh)
    axes.set_title("Input image")
    plt.show()
