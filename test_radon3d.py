from typing import Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as pgo
import time
import torch
import nrrd

import Extension as ExtensionTest

TaskSummaryRadon3D = Tuple[str, torch.Tensor]

size = [50, 50, 50]
spacing = torch.tensor([1., 1., 1.])
bounds = [0., 1.]

downsample_factor = 4


def task_radon3d(function, name: str, device: str, image: torch.Tensor) -> TaskSummaryRadon3D:
    image_devices = image.to(device=device)
    output = function(image_devices, spacing[0], spacing[1], spacing[2], size[0], size[1], size[2], 512)
    name: str = "{}_on_{}".format(name, device)
    return name, output.cpu()


def plot_task_radon3d(summary: TaskSummaryRadon3D):
    X, Y, Z = torch.meshgrid([torch.arange(0, size[0], 1), torch.arange(0, size[1], 1), torch.arange(0, size[2], 1)])
    fig = pgo.Figure(
        data=pgo.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=summary[1].flatten(), isomin=bounds[0],
                        isomax=bounds[1], opacity=.1, surface_count=21), layout=pgo.Layout(title=summary[0]))
    fig.show()


def benchmark_radon3d(path: str):
    print("----- Benchmarking radon3d -----")

    print("Loading CT data file {}...".format(path))
    data, header = nrrd.read(path)
    print("Done.")
    print("Processing CT data...")
    data = torch.tensor(data, device="cpu")
    image = torch.maximum(data.type(torch.float32) + 1000., torch.tensor([0.], device=data.device))
    down_sampler = torch.nn.AvgPool3d(downsample_factor)
    image = down_sampler(image[None, :, :, :])[0]
    global bounds
    bounds = [image.min().item(), image.max().item()]
    print("\tValue range = ({:.3f}, {:.3f})".format(bounds[0], bounds[1]))
    bounds[1] *= 10000.
    directions = torch.tensor(header['space directions'])
    global spacing
    spacing = float(downsample_factor) * directions.norm(dim=1)
    print("\tCT voxel spacing = [{} x {} x {}]".format(spacing[0], spacing[1], spacing[2]))
    print("Done.")
    # image = torch.tensor([
    #     [0.1, 0., 0.2, 0.4, 0.3, 0., .8],
    #     [0., 2., 0., 0.2, 0., 0., .8],
    #     [1., 0., 0.1, 0., 1., 0.4, .8],
    #     [0.1, 0.6, 0.4, 0.1, 0., 0.5, .8]
    # ])
    # image = torch.zeros((5, 5, 5))
    # image[0, 0, 0] = 1.
    # image[4, 3, 2] = .5

    outputs: list[TaskSummaryRadon3D] = []

    # print("Running on CPU...")
    # tic = time.time()
    # summary = task_radon3d(torch.ops.ExtensionTest.radon3d, "V1", "cpu", image)
    # toc = time.time()
    # print("Done; took {:.3f}s. Saving and plotting...".format(toc - tic))
    # torch.save(summary[1], "cache/{}.pt".format(summary[0]))
    # plot_task_radon3d(summary)
    # print("Done.")
    # outputs.append(summary)

    print("Running on CUDA...")
    tic = time.time()
    summary = task_radon3d(torch.ops.ExtensionTest.radon3d, "V1", "cuda", image)
    toc = time.time()
    print("Done; took {:.3f}s. Saving and plotting...".format(toc - tic))
    torch.save(summary[1], "cache/{}.pt".format(summary[0]))
    plot_task_radon3d(summary)
    print("Done.")
    outputs.append(summary)

    #
    # print("Running V2 on CPU...")
    # tic = time.time()
    # summary = task_radon3d(torch.ops.ExtensionTest.radon3d_v2, "V2", "cpu", image)
    # toc = time.time()
    # print("Done; took {:.3f}s. Saving and plotting...".format(toc - tic))
    # torch.save(summary[1], "cache/{}.pt".format(summary[0]))
    # plot_task_radon3d(summary)
    # print("Done.")
    # outputs.append(summary)
    #
    print("Running V2 on CUDA...")
    tic = time.time()
    summary = task_radon3d(torch.ops.ExtensionTest.radon3d_v2, "V2", "cuda", image)
    toc = time.time()
    print("Done; took {:.3f}s. Saving and plotting...".format(toc - tic))
    torch.save(summary[1], "cache/{}.pt".format(summary[0]))
    plot_task_radon3d(summary)
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

    # print("Showing plots...")
    # X, Y, Z = torch.meshgrid([torch.arange(0, size[0], 1), torch.arange(0, size[1], 1), torch.arange(0, size[2], 1)])
    # fig = pgo.Figure(
    #     data=pgo.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=image.flatten(), isomin=.0, isomax=2000.,
    #                     opacity=.1, surface_count=21), layout=pgo.Layout(title="Input"))
    # fig.show()
