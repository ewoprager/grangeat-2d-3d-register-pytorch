import itertools
import logging
import pathlib
import pprint
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
import pandas as pd
import scipy
import torch

from reg23_experiments.data.structs import Error, LinearRange
from reg23_experiments.utils.console_logging import tqdm

__all__ = ["experiments_hybrid", "experiments_cartesian", "experiments_sobol"]

logger = logging.getLogger(__name__)


def experiments_hybrid(  #
        param_constructor: Callable[[dict[str, Any]], Any | Error],  #
        experiment: Callable[[Any, torch.device, int, bool], pd.DataFrame | None],  #
        config_iterable: Iterable[tuple[str, dict[str, Any]]],  #
        output_directory: pathlib.Path,  #
        device: torch.device,  #
        tqdm_position: int = 0,  #
        dry_run: bool = False,  #
) -> None:
    assert output_directory.is_dir()
    tqdm_iterator = tqdm(  #
        config_iterable,  #
        desc="Experiments",  #
        position=tqdm_position,  #
        leave=None  #
    )
    for name, config in tqdm_iterator:
        # -----
        # Construct the experiment parameters
        parameters: Any | Error = param_constructor(config)
        if isinstance(parameters, Error):
            raise Exception(f"Failed to construct parameters at iteration {name}")
        # -----
        # Run the experiment
        try:
            res: pd.DataFrame | None = experiment(parameters, device, tqdm_position + 1, dry_run)
        except Exception as e:
            logger.error(f"Error running experiment at iteration {name}: {e}\nParameters:\n{pprint.pformat(config)}")
            continue
        if res is None:
            logger.info(f"Experiment at iteration {name}; configuration: \n{pprint.pformat(config)}\nwas deemed "
                        f"trivial / unnecessary.")
            continue
        # -----
        # Add the experiment config rows to the DataFrame and save
        if output_directory is not None:
            df = res.assign(**config)
            df.to_parquet(output_directory / f"data_{name}.parquet")


def experiments_cartesian(  #
        *,  #
        param_constructor: Callable[[dict[str, Any]], Any | Error],  #
        experiment: Callable[[Any, torch.device, int, bool], pd.DataFrame | None],  #
        params_to_vary: dict[str, list | torch.Tensor],  #
        constants: dict[str, Any],  #
        output_directory: pathlib.Path | None,  #
        device: torch.device,  #
        tqdm_position: int = 0,  #
        dry_run: bool = False,  #
) -> None:
    if output_directory is not None:
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
        if output_directory is not None:
            df = res.assign(**instance_all)
            df.to_parquet(output_directory / f"data_{"_".join([str(i) for i in indices])}.parquet")


def float01s_to_indices_linear(float01s: np.ndarray, index_count: int) -> np.ndarray:
    return np.minimum(index_count - 1, np.floor(float01s * float(index_count)).astype(int))


def experiments_sobol(  #
        *,  #
        m: int,  #
        param_constructor: Callable[[dict[str, Any]], Any | Error],  #
        experiment: Callable[[Any, torch.device, int, bool], pd.DataFrame | None],  #
        params_to_vary: dict[str, list | LinearRange],  #
        constants: dict[str, Any],  #
        output_directory: pathlib.Path,  #
        device: torch.device,  #
        tqdm_position: int = 0,  #
        dry_run: bool = False,  #
) -> None:
    """
    Run 2^m experiments where the parameters values are sampled from the given sets, generated by a Sobol sequence.
    :param m: The exponent of 2 that gives the number of samples to take
    :param param_constructor:
    :param experiment:
    :param params_to_vary: A dictionary mapping variable parameters to the sets of values they can take. Each set can
    be given as either a `list` of discrete values, or a `LinearRange` between which the values can be sampled.
    :param constants:
    :param output_directory:
    :param device:
    :param tqdm_position:
    :param dry_run:
    :return:
    """
    assert output_directory.is_dir()
    # -----
    # Determine the continuous values between 0 and 1 that correspond to the parameter values for each variable at
    # each iteration
    sobol_sequence = scipy.stats.qmc.Sobol(len(params_to_vary))
    all_float01s: np.ndarray = sobol_sequence.random_base2(m)  # size (2^m, len(params_to_vary))
    logger.info(f"Running experiments with the following constant parameters:\n{pprint.pformat(constants)}")
    # -----
    # Iterate through the rows of the parameters values, and run an experiment for each
    tqdm_iterator = tqdm(  #
        all_float01s,  #
        desc="Experiments",  #
        position=tqdm_position,  #
        leave=None  #
    )
    for i, float01s in enumerate(tqdm_iterator):
        # -----
        # Unpack the parameters for this iteration
        instance_specific: dict[str, Any] = {  #
            name: value_set.low + float01 * (value_set.high - value_set.low) if isinstance(value_set, LinearRange) else
            value_set[float01s_to_indices_linear(float01, len(value_set))]  #
            for float01, (name, value_set) in zip(float01s, params_to_vary.items())  #
        }  # config specific to this instance
        tqdm_iterator.set_postfix(**instance_specific)  # displaying this
        instance_all: dict[str, Any] = instance_specific | constants  # all the config for this instance
        # -----
        # Construct the experiment parameters
        parameters: Any | Error = param_constructor(instance_all)
        if isinstance(parameters, Error):
            raise Exception(f"Failed to construct parameters at float01s {float01s}")
        # -----
        # Run the experiment
        try:
            res: pd.DataFrame | None = experiment(parameters, device, tqdm_position + 1, dry_run)
        except Exception as e:
            logger.error(
                f"Error running experiment at float01s {float01s}: {e}\nParameters:\n{pprint.pformat(instance_all)}")
            continue
        if res is None:
            if not dry_run:
                logger.info(
                    f"Experiment at float01s {float01s}; configuration: \n{pprint.pformat(instance_specific)}\nwas "
                    f"deemed trivial / unnecessary.")
            continue
        # -----
        # Add the experiment config rows to the DataFrame and save
        df = res.assign(**instance_all)
        df.to_parquet(output_directory / f"data_sobol_{i}.parquet")
