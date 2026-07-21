import logging
import pathlib
from typing import Any

from reg23_app.cache_manager import CacheManager
from reg23_app.gui.drr_manager import DRRManager
from reg23_app.gui.fiducials_manager import FiducialsManager
from reg23_app.gui.input_manager import InputManager
from reg23_app.gui.reg_config_manager import RegConfigManager
from reg23_app.param_dadg_parity_manager import ParamDADGParityManager
from reg23_app.state import AppState
from reg23_experiments.data.ct_fiducial_save_data import CTFiducialSaveManager
from reg23_experiments.data.electrode_save_data import ElectrodeSaveManager
from reg23_experiments.data.parameters import Parameters
from reg23_experiments.data.transformation_save_data import TransformationSaveManager
from reg23_experiments.data.xray_fiducial_save_data import XRayFiducialSaveManager
from reg23_experiments.data.xray_reg_save_data import XRayRegSaveManager
from reg23_experiments.io.serialize import deserialize_recursive, serialize_recursive
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph
from reg23_experiments.utils.data import observe_all_traits_recursively

__all__ = ["AppContext"]

logger = logging.getLogger(__name__)


class AppContext:
    """
    The `AppContext` owns and provides access to all 'global' resources, and is passed around to most things.

    Most notably:
    - The app state object, which contains just simple, serializable data
    - The DADG, which manages all complex and heavy data and the relationships between them
    - Objects that manage saving of data to caches / save directories

    Also manages automatic loading of values from the cache into the state on initialisation, and forwarding appropriate
    changes of the state to the cache manager for eagerly saving to the cache.
    """

    def __init__(  #
            self,  #
            *,  #
            parameters: Parameters,  #
            dadg: DirectedAcyclicDataGraph,  #
            electrode_save_directory: pathlib.Path,  #
            transformation_save_directory: pathlib.Path,  #
            ct_fiducial_save_directory: pathlib.Path,  #
            xray_fiducial_save_directory: pathlib.Path,  #
            xray_reg_save_directory: pathlib.Path,  #
            cache: bool = True  #
    ):
        logger.debug("Initialising AppContext")
        self._input_manager = InputManager()
        self._state = AppState(parameters=parameters)
        self._dadg = dadg
        self._electrode_save_manager = ElectrodeSaveManager(electrode_save_directory)
        self._transformation_save_manager = TransformationSaveManager(transformation_save_directory)
        self._ct_fiducial_save_manager = CTFiducialSaveManager(ct_fiducial_save_directory)
        self._xray_fiducial_save_manager = XRayFiducialSaveManager(xray_fiducial_save_directory)
        self._xray_reg_save_manager = XRayRegSaveManager(xray_reg_save_directory)
        self._drr_manager = DRRManager(self._state, self._dadg)
        self._fiducials_manager = FiducialsManager(self._state, self._dadg)
        self._cache = cache

        if self._cache:
            self._cache_manager = CacheManager()
            # load params from the cache
            res: dict[str, Any] | None = self._cache_manager.last_params
            if res is not None:
                self.state.parameters = deserialize_recursive(value=res, old_value=self.state.parameters,
                                                              trait=self.state.traits()["parameters"])

        self._param_dadg_parity_manager = ParamDADGParityManager(  #
            state=self._state, dadg=self._dadg, electrode_save_manager=self._electrode_save_manager,
            ct_fiducial_save_manager=self._ct_fiducial_save_manager,
            xray_fiducial_save_manager=self._xray_fiducial_save_manager)
        self._reg_config_manager = RegConfigManager(self._state, self._dadg,
                                                    xray_reg_save_manager=self._xray_reg_save_manager)

        # observing all the parameter widgets
        observe_all_traits_recursively(self._any_parameter_changed, self._state.parameters)

    @property
    def input_manager(self) -> InputManager:
        return self._input_manager

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def dadg(self) -> DirectedAcyclicDataGraph:
        return self._dadg

    @property
    def electrode_save_manager(self) -> ElectrodeSaveManager:
        return self._electrode_save_manager

    @property
    def transformation_save_manager(self) -> TransformationSaveManager:
        return self._transformation_save_manager

    @property
    def ct_fiducial_save_manager(self) -> CTFiducialSaveManager:
        return self._ct_fiducial_save_manager

    @property
    def xray_fiducial_save_manager(self) -> XRayFiducialSaveManager:
        return self._xray_fiducial_save_manager

    def _any_parameter_changed(self, change) -> None:
        if self._cache:
            self._cache_manager.last_params = serialize_recursive(self.state.parameters,
                                                                  trait=self.state.traits()["parameters"])
