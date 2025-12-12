import traitlets
import logging
import inspect

from program.lib.structs import Error, FunctionArgument
from program import data_manager
from program.lib import optimisation

logger = logging.getLogger(__name__)


class OptimisationStage(traitlets.HasTraits):
    name = traitlets.Unicode()
    params_set = traitlets.Dict(key_trait=traitlets.Unicode())


def run_optimisation(optimisation_stage: OptimisationStage) -> optimisation.OptimisationResult | Error:
    logger.info(f"Initialising for optimisation stage '{optimisation_stage.name}'...")
    err = data_manager().set_data(**optimisation_stage.params_set)
    if isinstance(err, Error):
        return Error(f"Error setting params for optimisation stage '{optimisation_stage.name}': {err.description}.")
    logger.info(f"Running optimisation stage '{optimisation_stage.name}'...")
    res = data_manager().get("optimisation_algorithm")
    if isinstance(err, Error):
        return Error(
            f"Error getting variable 'optimisation_algorithm) for optimisation stage '{optimisation_stage.name}': "
            f"{err.description}.")
    return res()


class Registration(traitlets.HasTraits):
    name = traitlets.Unicode()
    stages = traitlets.List(trait=traitlets.Instance(OptimisationStage))


def run_registration(registration: Registration) -> None | Error:
    logger.info(f"Running registration '{registration.name}'...")
    for stage in registration.stages:
        res = run_optimisation(stage)
        if isinstance(res, Error):
            return Error(f"Error running stage for registration '{registration.name}': {res.description}.")
    return None
