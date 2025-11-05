from typing import Tuple, NamedTuple
import time

import torch
import matplotlib.pyplot as plt
import numpy as np

import reg23

from notification import logs_setup
import registration.objective_function as objective_function

TaskSummarySimilarity = Tuple[str, torch.Tensor, float]


def zncc_torch(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    return torch.corrcoef(torch.stack((xs.flatten(), ys.flatten()), dim=0))[0, 1]


def task_similarity(function, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    for _ in range(100):
        output = function(a, b)
    return output


def plot_task_similarity(summary: TaskSummarySimilarity):
    logger.info("{}: {}".format(summary[0], summary[1].item()))


def task_autograd(function, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    for _ in range(100):
        this_a = a.clone()
        this_a.requires_grad = True
        output = function(this_a, b)
        output.backward()
        grad = this_a.grad
    return grad


def plot_task_autograd(summary: TaskSummarySimilarity):
    logger.info("{}: {}".format(summary[0], summary[1]))


def run_task(task, task_plot, function, name: str, device: str, a: torch.Tensor,
             b: torch.Tensor) -> TaskSummarySimilarity:
    a_device = a.to(device=device)
    b_device = b.to(device=device)
    logger.info("Running {} on {}...".format(name, device))
    torch.cuda.synchronize()
    tic = time.time()
    output = task(function, a_device, b_device)
    torch.cuda.synchronize()
    toc = time.time()
    elapsed: float = toc - tic
    logger.info("Done; took {:.3f}s. Saving and plotting...".format(elapsed))
    name: str = "{}_on_{}".format(name, device)
    summary = name, output.cpu(), elapsed
    torch.save(summary[1], "cache/{}.pt".format(summary[0]))
    task_plot(summary)
    logger.info("Done.")
    return summary


def main():
    logger.info("----- Benchmarking normalised_cross_correlation -----")

    image_size = torch.Size([1000, 1000])

    a = torch.rand(image_size)
    b = torch.rand(image_size)

    outputs: list[TaskSummarySimilarity] = [  #
        run_task(task_similarity, plot_task_similarity, objective_function.ncc, "ZNCC python", "cpu", a, b),  #
        run_task(task_similarity, plot_task_similarity, reg23.normalised_cross_correlation, "ZNCC C++", "cpu", a, b),  #
        run_task(task_similarity, plot_task_similarity, zncc_torch, "PyTorch corrcoef", "cpu", a, b)  #
    ]

    if torch.cuda.is_available():
        outputs += [  #
            run_task(task_similarity, plot_task_similarity, objective_function.ncc, "ZNCC python", "cuda", a, b),  #
            run_task(task_similarity, plot_task_similarity, reg23.normalised_cross_correlation, "ZNCC CUDA", "cuda", a,
                     b),  #
            run_task(task_similarity, plot_task_similarity, zncc_torch, "PyTorch corrcoef", "cuda", a, b)  #

        ]

    logger.info("Calculating discrepancies...")
    found: bool = False
    for i in range(len(outputs) - 1):
        discrepancy = ((outputs[i][1] - outputs[i + 1][1]).abs() / (
                .5 * (outputs[i][1] + outputs[i + 1][1]).abs() + 1e-5)).mean()
        if discrepancy > 1e-2:
            found = True
            logger.info("\tAverage discrepancy between outputs {} and {} is {:.3f} %"
                        "".format(outputs[i][0], outputs[i + 1][0], 100. * discrepancy))
    if not found:
        logger.info("\tNo discrepancies found.")
    logger.info("Done.")

    function_names = ["ZNCC Python", "ZNCC C++/CUDA", "torch.corrcoef"]
    x = np.arange(len(function_names))
    width = 0.25
    fig, axes = plt.subplots()
    for i, device_name in enumerate(["CPU", "CUDA"]):
        offset = width * float(i)
        rects = axes.bar(x + offset, [output[2] for output in outputs[(3 * i):(3 * (i + 1))]], width, label=device_name)
        axes.bar_label(rects, padding=3)
    axes.set_xticks(x + width, function_names)
    axes.legend()
    axes.set_ylabel("Time taken for 100 evaluations [s]")
    axes.set_title("ZNCC")

    # Benchmarking autograd
    logger.info("-----")
    logger.info("Benchmarking autograd")
    logger.info("-----")

    outputs_autograd: list[TaskSummarySimilarity] = [  #
        run_task(task_autograd, plot_task_autograd, objective_function.ncc, "ZNCC python", "cpu", a, b),  #
        run_task(task_autograd, plot_task_autograd, reg23.autograd.normalised_cross_correlation, "ZNCC C++", "cpu", a,
                 b),  #
        run_task(task_autograd, plot_task_autograd, zncc_torch, "PyTorch corrcoef", "cpu", a, b)  #
    ]

    if torch.cuda.is_available():
        outputs_autograd += [  #
            run_task(task_autograd, plot_task_autograd, objective_function.ncc, "ZNCC python", "cuda", a, b),  #
            run_task(task_autograd, plot_task_autograd, reg23.autograd.normalised_cross_correlation, "ZNCC CUDA",
                     "cuda", a, b),  #
            run_task(task_autograd, plot_task_autograd, zncc_torch, "PyTorch corrcoef", "cuda", a, b)  #

        ]

    logger.info("Calculating discrepancies...")
    found: bool = False
    for i in range(len(outputs_autograd) - 1):
        discrepancy = ((outputs_autograd[i][1] - outputs_autograd[i + 1][1]).abs() / (
                .5 * (outputs_autograd[i][1] + outputs_autograd[i + 1][1]).abs() + 1e-5)).mean()
        if discrepancy > 1e-2:
            found = True
            logger.info("\tAverage discrepancy between autograd outputs {} and {} is {:.3f} %"
                        "".format(outputs_autograd[i][0], outputs_autograd[i + 1][0], 100. * discrepancy))
    if not found:
        logger.info("\tNo discrepancies found.")
    logger.info("Done.")

    function_names = ["ZNCC Python", "ZNCC C++/CUDA", "torch.corrcoef"]
    x = np.arange(len(function_names))
    width = 0.25
    fig, axes = plt.subplots()
    for i, device_name in enumerate(["CPU", "CUDA"]):
        offset = width * float(i)
        rects = axes.bar(x + offset, [output[2] for output in outputs_autograd[(3 * i):(3 * (i + 1))]], width,
                         label=device_name)
        axes.bar_label(rects, padding=3)
    axes.set_xticks(x + width, function_names)
    axes.legend()
    axes.set_ylabel("Time taken for 100 evaluations [s]")
    axes.set_title("ZNCC with autograd and single backward pass")

    plt.show()

    # logger.info("Showing plots...")  # X, Y, Z = torch.meshgrid([torch.arange(0, size[0], 1), torch.arange(0,  # size[1], 1), torch.arange(0, size[2], 1)])  # fig = pgo.Figure(  #     data=pgo.Volume(x=X.flatten(),  # y=Y.flatten(), z=Z.flatten(), value=image.flatten(), isomin=.0, isomax=2000.,  #  # opacity=.1,  # surface_count=21), layout=pgo.Layout(title="Input"))  # fig.show()


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    main()
