from typing import Tuple

import matplotlib.cm
import matplotlib.pyplot as plt
import time
import torch
import pydicom

import Extension as ExtensionTest

TaskSummaryRadon2D = Tuple[str, torch.Tensor]

downsample_factor = 1
spacing = [1., 1.]

def task_radon2d(function, name: str, device: str, image: torch.Tensor) -> TaskSummaryRadon2D:
    image_devices = image.to(device=device)
    output = function(image_devices, spacing[0], spacing[1], 200, 200, 512)
    return "{} on {}".format(name, device), output.cpu()


def plot_task_radon2d(summary: TaskSummaryRadon2D):
    fig, axes = plt.subplots()
    mesh = axes.pcolormesh(summary[1])
    fig.colorbar(mesh)
    axes.set_title(summary[0])


def run_task(task, task_plot, function, name: str, device: str, image: torch.Tensor):
    print("Running {} on {}...".format(name, device))
    tic = time.time()
    summary = task(function, name, device, image)
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

    print("Loading X-ray data file {}...".format(path))
    x_ray = pydicom.dcmread(path)
    print("Done.")
    print("Processing X-ray data...")
    data = torch.tensor(x_ray.pixel_array, device="cpu")
    sizes = data.size()
    print("\tX-ray size = [{} x {}]".format(sizes[0], sizes[1]))
    image = torch.maximum(data.type(torch.float32), torch.tensor([0.], device=data.device))
    global downsample_factor
    if downsample_factor > 1:
        down_sampler = torch.nn.AvgPool2d(downsample_factor)
        image = down_sampler(image[None, :, :])[0]
    else:
        downsample_factor = 1
    sizes = image.size()
    print("\tX-ray size after down-sampling = [{} x {}]".format(sizes[0], sizes[1]))
    global bounds
    bounds = [image.min().item(), image.max().item()]
    print("\tValue range = ({:.3f}, {:.3f})".format(bounds[0], bounds[1]))
    # bounds[1] *= 10000.
    directions = torch.tensor(x_ray.PixelSpacing)
    global spacing
    spacing = float(downsample_factor) * directions
    print("\tX-ray pixel spacing = [{} x {}] mm".format(spacing[0], spacing[1]))
    print("Done.")

    outputs: list[TaskSummaryRadon2D] = [
        run_task(task_radon2d, plot_task_radon2d, ExtensionTest.radon2d, "V1", "cpu", image),
        run_task(task_radon2d, plot_task_radon2d, ExtensionTest.radon2d, "V1", "cuda", image),
        run_task(task_radon2d, plot_task_radon2d, ExtensionTest.radon2d_v2, "V2", "cpu", image),
        run_task(task_radon2d, plot_task_radon2d, ExtensionTest.radon2d_v2, "V2", "cuda", image)]

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

    # image = torch.tensor(
    #     [[0.1, 0., 0.2, 0.4, 0.3, 0., .8], [0., 2., 0., 0.2, 0., 0., .8], [1., 0., 0.1, 0., 1., 0.4, .8],
    #      [0.1, 0.6, 0.4, 0.1, 0., 0.5, .8]])



    outputs: list[TaskSummaryRadon2D] = [
        run_task(task_radon2d, plot_task_radon2d, ExtensionTest.dRadon2dDR, "V1", "cpu", image)]

    # print("Calculating discrepancies...")
    # found: bool = False
    # for i in range(len(outputs) - 1):
    #     discrepancy = ((outputs[i][1] - outputs[i + 1][1]).abs() / (
    #             .5 * (outputs[i][1] + outputs[i + 1][1]).abs() + 1e-5)).mean()
    #     if discrepancy > 1e-2:
    #         found = True
    #         print("\tAverage discrepancy between outputs {} and {} is {:.3f} %".format(outputs[i][0], outputs[i + 1][0],
    #                                                                                    100. * discrepancy))
    # if not found:
    #     print("\tNo discrepancies found.")
    # print("Done.")

    print("Showing plots...")
    fig, axes = plt.subplots()
    mesh = axes.pcolormesh(image)
    fig.colorbar(mesh)
    axes.set_title("Input image")
    plt.show()
