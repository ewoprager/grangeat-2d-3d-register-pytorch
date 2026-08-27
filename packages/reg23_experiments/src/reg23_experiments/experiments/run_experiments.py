import logging
import pathlib
import pprint
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

import pandas as pd
import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.utils.console_logging import indentation_prefix, tqdm

__all__ = ["Experiment", "ExperimentConfig", "run_experiments"]

logger = logging.getLogger(__name__)

type Experiment[T] = Callable[[ExperimentConfig[T]], pd.DataFrame | None]
"""
A function that can be run as an experiment.
Generic over the type T of the experiment's parametrisation; this would likely want to be a HasTraits class that can be
constructed from a dictionary of parameter values.
"""


class ExperimentConfig[T](NamedTuple):
    """
    A struct containing the necessary parameters for an experiment.
    Generic over the type T of the experiment's parametrisation; this would likely want to be a HasTraits class that can
    be constructed from a dictionary of parameter values.
    """
    params: T
    device: torch.device
    tqdm_position: int
    dry_run: bool


def run_experiments[T](  #
        *,  #
        param_constructor: Callable[[dict[str, Any]], T | Error],  #
        experiment: Experiment[T],  #
        parametrisation_iterable: Iterable[tuple[str, dict[str, Any]]],  #
        output_directory: pathlib.Path | None,  #
        device: torch.device,  #
        tqdm_position: int = 0,  #
        dry_run: bool = False,  #
        throw: bool = False,  #
        overwrite: bool = False,  #
) -> None:
    """
    Iterate through the iterable of experiment parametrisations, run the given experiment for each one, and save the
    results of each experiment as a separate output file in the given directory.

    Generic over the type T of the experiment's parametrisation; this would likely want to be a HasTraits class that can
    be constructed from a dictionary of parameter values.

    This function uses tqdm to neatly display the progress of the experiments.

    :param param_constructor: A function that maps a dictionary of parameter values to an experiment parametrisation.
    :param experiment: An experiment function.
    :param parametrisation_iterable: An iterable that returns dictionaries of experiment parametrisations.
    :param output_directory: The directory into which to save the experiment output files.
    :param device: The torch device to use.
    :param tqdm_position: The vertical position of the tqdm progress bar in the output; only need to be set if this
    function is run within a tqdm progress loop, in which case the position should be 1 greater than the previous.
    :param dry_run: Whether to pass the 'dry_run' flag on to the experiments within. Also modifies throwing behaviour.
    :param throw: Whether to throw on failure of an experiment. If not, exceptions raised by experiments will be
    caught so that the other experiments may continue.
    :param overwrite: Whether to overwrite experiment output files in the output directory if they already exist.
    """
    if output_directory is not None:
        assert output_directory.is_dir()
    tqdm_iterator = tqdm(  #
        parametrisation_iterable,  #
        desc=indentation_prefix(tqdm_position) + "Dry run of experiments" if dry_run else "Experiments",  #
        position=tqdm_position,  #
        leave=None  #
    )
    for name, parametrisation in tqdm_iterator:
        tqdm_iterator.set_postfix({"iteration": name})
        # -----
        # Skip if this results file already exists, and not configured to overwrite
        if output_directory is None:
            output_file = None
        else:
            output_file = output_directory / f"data_{name}.parquet"
            if not overwrite and output_file.exists():
                logger.info(f"Skipping experiment '{name}' as results file '{str(output_file)}' already exists.")
                continue
        # -----
        # Construct the experiment parameters
        parameters: T | Error = param_constructor(parametrisation)
        if isinstance(parameters, Error):
            raise Exception(f"Failed to construct parameters at iteration {name}")
        # -----
        # Run the experiment
        config: ExperimentConfig[T] = ExperimentConfig(  #
            params=parameters,  #
            device=device,  #
            tqdm_position=tqdm_position + 1,  #
            dry_run=dry_run,  #
        )
        if throw or dry_run:
            res: pd.DataFrame | None = experiment(config)
        else:
            try:
                res: pd.DataFrame | None = experiment(config)
            except Exception as e:
                logger.error(
                    f"Error running experiment at iteration {name}: {e}\nParameters:\n"
                    f"{pprint.pformat(parametrisation)}")
                continue
        if dry_run:
            continue
        if res is None:
            logger.info(
                f"Experiment at iteration {name}; configuration: \n{pprint.pformat(parametrisation)}\nwas deemed "
                f"trivial / unnecessary.")
            continue
        # -----
        # Add the experiment config rows to the DataFrame and save
        if output_file is not None:
            df = res.assign(**parametrisation)
            df.to_parquet(output_file)
