import time
import torch
import matplotlib.pyplot as plt
from typing import Tuple

import Extension as ExtensionTest

TaskSummary = Tuple[str, torch.Tensor]


def task(function, name: str, device: str, image: torch.Tensor) -> TaskSummary:
    image_devices = image.to(device=device)
    output = function(image_devices, 100, 100, 1024)
    return "{} on {}".format(name, device), output.cpu()


def plot_task(summary: TaskSummary):
    _, axes = plt.subplots()
    axes.pcolormesh(summary[1])
    axes.set_title(summary[0])


if __name__ == "__main__":
    print("----- Benchmarking radon2d -----")

    image = torch.tensor([
        [0.1, 0., 0.2, 0.4, 0.3, 0., .8],
        [0., 2., 0., 0.2, 0., 0., .8],
        [1., 0., 0.1, 0., 1., 0.4, .8],
        [0.1, 0.6, 0.4, 0.1, 0., 0.5, .8]
    ])

    outputs: list[torch.Tensor] = []

    print("Running on CPU...")
    tic = time.time()
    summary = task(torch.ops.ExtensionTest.radon2d, "V1", "cpu", image)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    plot_task(summary)
    print("Done.")
    outputs.append(summary[1])

    print("Running on CUDA...")
    tic = time.time()
    summary = task(torch.ops.ExtensionTest.radon2d, "V1", "cuda", image)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    plot_task(summary)
    print("Done.")
    outputs.append(summary[1])

    print("Running V2 on CPU...")
    tic = time.time()
    summary = task(torch.ops.ExtensionTest.radon2d_v2, "V2", "cpu", image)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    plot_task(summary)
    print("Done.")
    outputs.append(summary[1])

    print("Running V2 on CUDA...")
    tic = time.time()
    summary = task(torch.ops.ExtensionTest.radon2d_v2, "V2", "cuda", image)
    toc = time.time()
    print("Done; took {:.3f}s. Plotting summary...".format(toc - tic))
    plot_task(summary)
    print("Done.")
    outputs.append(summary[1])

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
    _, axes = plt.subplots()
    axes.pcolormesh(image)
    axes.set_title("Input image")
    plt.show()
