from typing import Tuple, NamedTuple
import time

import torch
import matplotlib.pyplot as plt

import Extension as reg23

import logs_setup
import registration.objective_function as objective_function

TaskSummarySimilarity = Tuple[str, torch.Tensor]


def zncc_torch(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    return torch.corrcoef(torch.stack((xs.flatten(), ys.flatten()), dim=0))[0, 1]


def task_similarity(function, a: torch.Tensor, b: torch.Tensor) -> TaskSummarySimilarity:
    for _ in range(100):
        output = function(a, b)
    return output


def plot_task_similarity(summary: TaskSummarySimilarity):
    logger.info("{}: {}".format(summary[0], summary[1].item()))


def run_task(task, task_plot, function, name: str, device: str, a: torch.Tensor,
             b: torch.Tensor) -> TaskSummarySimilarity:
    a_device = a.to(device=device)
    b_device = b.to(device=device)
    logger.info("Running {} on {}...".format(name, device))
    tic = time.time()
    output = task(function, a_device, b_device)
    toc = time.time()
    logger.info("Done; took {:.3f}s. Saving and plotting...".format(toc - tic))
    name: str = "{}_on_{}".format(name, device)
    summary = name, output.cpu()
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
        run_task(task_similarity, plot_task_similarity, objective_function.ncc, "ZNCC", "cpu", a, b),
        run_task(task_similarity, plot_task_similarity, reg23.normalised_cross_correlation,
                 "NormalisedCrossCorrelation", "cpu", a, b),
        run_task(task_similarity, plot_task_similarity, zncc_torch, "ZNCC torch", "cpu", a, b)  #
    ]

    if torch.cuda.is_available():
        outputs += [  #
            run_task(task_similarity, plot_task_similarity, objective_function.ncc, "ZNCC", "cuda", a, b),
            run_task(task_similarity, plot_task_similarity, reg23.normalised_cross_correlation,
                     "NormalisedCrossCorrelation", "cuda", a, b)  #
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

    # logger.info("Showing plots...")  # X, Y, Z = torch.meshgrid([torch.arange(0, size[0], 1), torch.arange(0,  # size[1], 1), torch.arange(0, size[2], 1)])  # fig = pgo.Figure(  #     data=pgo.Volume(x=X.flatten(),  # y=Y.flatten(), z=Z.flatten(), value=image.flatten(), isomin=.0, isomax=2000.,  #  # opacity=.1,  # surface_count=21), layout=pgo.Layout(title="Input"))  # fig.show()


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    main()
