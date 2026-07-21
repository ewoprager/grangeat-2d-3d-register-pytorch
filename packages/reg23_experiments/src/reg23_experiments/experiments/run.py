import itertools
import logging
import pathlib
import pprint
from typing import Any, Callable

import matplotlib

matplotlib.use("QtAgg")

import pandas as pd
import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.utils.console_logging import tqdm

__all__ = ["experiments_cartesian"]

logger = logging.getLogger(__name__)


def experiments_cartesian(  #
        *,  #
        param_constructor: Callable[[dict[str, Any]], Any | Error],  #
        experiment: Callable[[Any, torch.device, int, bool], pd.DataFrame | None],  #
        params_to_vary: dict[str, list | torch.Tensor],  #
        constants: dict[str, Any],  #
        output_directory: pathlib.Path,  #
        device: torch.device,  #
        tqdm_position: int = 0,  #
        dry_run: bool = False,#
) -> None:
    assert output_directory.is_dir()
    # -----
    # Determine the total number of experiments being run
    each_range_length = []
    for name, values in params_to_vary.items():
        if isinstance(values, torch.Tensor):
            assert len(values.size()) == 1
        each_range_length.append(len(values))
    total = 1
    for l in each_range_length:
        total *= l
    logger.info(f"Running experiments with the following constant parameters:\n{pprint.pformat(constants)}")
    # -----
    # Iterate through the Cartesian product of the sets of parameters values, and run an experiment for each
    tqdm_iterator = tqdm(  #
        itertools.product(*(range(l) for l in each_range_length)),  #
        desc="Experiments",  #
        total=total,  #
        position=tqdm_position,  #
        leave=None  #
    )
    for indices in tqdm_iterator:
        # -----
        # Unpack the parameters for this iteration
        instance_specific: dict[str, Any] = {  #
            name: values[index]  #
            for index, (name, values) in zip(indices, params_to_vary.items())  #
        }  # config specific to this instance
        tqdm_iterator.set_postfix(**instance_specific)  # displaying this
        instance_all: dict[str, Any] = instance_specific | constants  # all the config for this instance
        # -----
        # Construct the experiment parameters
        parameters: Any | Error = param_constructor(instance_all)
        if isinstance(parameters, Error):
            raise Exception(f"Failed to construct parameters at indices {indices}")
        # -----
        # Run the experiment
        try:
            res: pd.DataFrame | None = experiment(parameters, device, tqdm_position + 1, dry_run)
        except Exception as e:
            logger.error(
                f"Error running experiment at indices {indices}: {e}\nParameters:\n{pprint.pformat(instance_all)}")
            continue
        if res is None:
            logger.info(
                f"Experiment at indices {indices}; configuration: \n{pprint.pformat(instance_specific)}\nwas deemed "
                f"trivial / unnecessary.")
            continue
        # -----
        # Add the experiment config rows to the DataFrame and save
        df = res.assign(**instance_all)
        df.to_parquet(output_directory / f"data_{"_".join([str(i) for i in indices])}.parquet")
