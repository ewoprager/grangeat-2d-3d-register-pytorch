from typing import Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as pgo
import time
import torch

import Extension as ExtensionTest

TaskSummaryRadon3D = Tuple[str, torch.Tensor]

size = [30, 30, 30]


def task_radon3d(function, name: str, device: str, image: torch.Tensor) -> TaskSummaryRadon3D:
    image_devices = image.to(device=device)
    output = function(image_devices, *size, 128)
    return "{} on {}".format(name, device), output.cpu()


def plot_task_radon3d(summary: TaskSummaryRadon3D):
    X, Y, Z = torch.meshgrid([torch.arange(0, size[0], 1), torch.arange(0, size[1], 1), torch.arange(0, size[2], 1)])
    fig = pgo.Figure(
        data=pgo.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=summary[1].flatten(), isomin=.0, isomax=2.,
                        opacity=.1, surface_count=21))
    fig.show()


def benchmark_radon3d():
    print("----- Benchmarking radon3d -----")

    # image = torch.tensor([
    #     [0.1, 0., 0.2, 0.4, 0.3, 0., .8],
    #     [0., 2., 0., 0.2, 0., 0., .8],
    #     [1., 0., 0.1, 0., 1., 0.4, .8],
    #     [0.1, 0.6, 0.4, 0.1, 0., 0.5, .8]
    # ])
    image = torch.zeros((5, 5, 5))
    image[0, 0, 0] = 1.
    image[4, 3, 2] = .5

    outputs: list[torch.Tensor] = []

    print("Running on CPU...")
    tic = time.time()
    summary = task_radon3d(torch.ops.ExtensionTest.radon3d, "V1", "cpu", image)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    plot_task_radon3d(summary)
    print("Done.")
    outputs.append(summary[1])

    # print("Running on CUDA...")
    # tic = time.time()
    # summary = task_radon3d(torch.ops.ExtensionTest.radon2d, "V1", "cuda", image)
    # toc = time.time()
    # print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    # plot_task_radon3d(summary)
    # print("Done.")
    # outputs.append(summary[1])
    #
    # print("Running V2 on CPU...")
    # tic = time.time()
    # summary = task_radon3d(torch.ops.ExtensionTest.radon2d_v2, "V2", "cpu", image)
    # toc = time.time()
    # print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    # plot_task_radon3d(summary)
    # print("Done.")
    # outputs.append(summary[1])
    #
    # print("Running V2 on CUDA...")
    # tic = time.time()
    # summary = task_radon3d(torch.ops.ExtensionTest.radon2d_v2, "V2", "cuda", image)
    # toc = time.time()
    # print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    # plot_task_radon3d(summary)
    # print("Done.")
    # outputs.append(summary[1])

    print("Calculating discrepancies...")
    found: bool = False
    for i in range(len(outputs) - 1):
        discrepancy = ((outputs[i] - outputs[i + 1]).abs() / outputs[i].abs()).mean()
        if discrepancy > 1e-5:
            found = True
            print("Average discrepancy between outputs {} and {} is {:.4e} %".format(i, i + 1, 100. * discrepancy))
    if not found:
        print("No discrepancies found.")
    print("Done.")

    print("Showing plots...")
    X, Y, Z = torch.meshgrid([torch.arange(0, size[0], 1), torch.arange(0, size[1], 1), torch.arange(0, size[2], 1)])
    fig = pgo.Figure(
        data=pgo.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=image.flatten(), isomin=.0, isomax=2.,
                        opacity=.1, surface_count=21))
    fig.show()
