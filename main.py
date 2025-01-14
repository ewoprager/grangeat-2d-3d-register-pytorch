import time
import torch
import matplotlib.pyplot as plt
from typing import Tuple

import Extension as ExtensionTest


TaskSummary = Tuple[str, torch.Tensor]


def task(function, name: str, device: str, image: torch.Tensor) -> TaskSummary:
    # image_devices = torch.tensor(image, device=device)
    image_devices = torch.tensor([
        [1.1, 0., 0.2, 0.4, 0.3, 0., .8],
        [0., 2., 0., 0.2, 0., 0., .8],
        [1., 0., .1, 0., 1., 0.4, .8],
        [0.1, 0.6, 0.4, 0.1, 0., 0.5, .8]
    ], device=device)
    output = function(image_devices, 1000, 1000, 256)
    return "{} on {}".format(name, device), output


def plot_task(summary: TaskSummary):
    _, axes = plt.subplots()
    axes.pcolormesh(summary[1].cpu())
    axes.set_title(summary[0])


if __name__ == "__main__":
    print("----- Benchmarking radon2d -----")

    image = torch.tensor([
        [0.1, 0., 0.2, 0.4, 0.3, 0., .8],
        [0., 2., 0., 0.2, 0., 0., .8],
        [1., 0., 0.1, 0., 1., 0.4, .8],
        [0.1, 0.6, 0.4, 0.1, 0., 0.5, .8]
    ])

    print("Running on CPU...")
    tic = time.time()
    summary = task(torch.ops.ExtensionTest.radon2d, "V1", "cpu", image)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    plot_task(summary)
    print("Done.")

    print("Running on CUDA...")
    tic = time.time()
    summary = task(torch.ops.ExtensionTest.radon2d, "V1", "cuda", image)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    plot_task(summary)
    print("Done.")

    print("Running V2 on CPU...")
    tic = time.time()
    summary = task(torch.ops.ExtensionTest.radon2d_v2, "V2", "cpu", image)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    plot_task(summary)
    print("Done.")

    print("Running V2 on CUDA...")
    tic = time.time()
    summary = task(torch.ops.ExtensionTest.radon2d_v2, "V2", "cuda", image)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    plot_task(summary)
    print("Done.")

    print("Showing plots...")
    _, axes = plt.subplots()
    axes.pcolormesh(image)
    axes.set_title("Input image")
    plt.show()

