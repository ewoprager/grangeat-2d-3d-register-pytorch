from typing import Tuple

import matplotlib.cm
import matplotlib.pyplot as plt
import time
import torch

import Extension as ExtensionTest

TaskSummaryRadon2D = Tuple[str, torch.Tensor]


def task_radon2d(function, name: str, device: str, image: torch.Tensor) -> TaskSummaryRadon2D:
    image_devices = image.to(device=device)
    output = function(image_devices, 1., 1., 200, 200, 4096)
    return "{} on {}".format(name, device), output.cpu()


def plot_task_radon2d(summary: TaskSummaryRadon2D):
    fig, axes = plt.subplots()
    mesh = axes.pcolormesh(summary[1])
    fig.colorbar(mesh)
    axes.set_title(summary[0])


def benchmark_radon2d():
    print("----- Benchmarking radon2d -----")

    image = torch.tensor(
        [[0.1, 0., 0.2, 0.4, 0.3, 0., .8], [0., 2., 0., 0.2, 0., 0., .8], [1., 0., 0.1, 0., 1., 0.4, .8],
            [0.1, 0.6, 0.4, 0.1, 0., 0.5, .8]])

    outputs: list[TaskSummaryRadon2D] = []

    print("Running on CPU...")
    tic = time.time()
    summary = task_radon2d(torch.ops.ExtensionTest.radon2d, "V1", "cpu", image)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    plot_task_radon2d(summary)
    print("Done.")
    outputs.append(summary)

    print("Running on CUDA...")
    tic = time.time()
    summary = task_radon2d(torch.ops.ExtensionTest.radon2d, "V1", "cuda", image)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    plot_task_radon2d(summary)
    print("Done.")
    outputs.append(summary)

    print("Running V2 on CPU...")
    tic = time.time()
    summary = task_radon2d(torch.ops.ExtensionTest.radon2d_v2, "V2", "cpu", image)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    plot_task_radon2d(summary)
    print("Done.")
    outputs.append(summary)

    print("Running V2 on CUDA...")
    tic = time.time()
    summary = task_radon2d(torch.ops.ExtensionTest.radon2d_v2, "V2", "cuda", image)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    plot_task_radon2d(summary)
    print("Done.")
    outputs.append(summary)

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
