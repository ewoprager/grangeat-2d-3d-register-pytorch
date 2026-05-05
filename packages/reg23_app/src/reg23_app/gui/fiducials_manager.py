import logging

import torch
import numpy as np
import scipy as sp

from reg23_app.state import AppState

from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation, \
    mapping_transformation_to_parameters

__all__ = ["FiducialsManager"]

logger = logging.getLogger(__name__)


class FiducialsManager:
    """
    No widgets

    Reads from and write to the state and DADG only
    """

    def __init__(self, state: AppState, dadg: DirectedAcyclicDataGraph):
        self._state = state
        self._dadg = dadg

        self._state.observe(self._button_fiducial_register, names=["button_fiducial_register"])

    def _button_fiducial_register(self, change) -> None:
        if not change.new:
            return
        self._state.button_fiducial_register = False

        if self._state.register_fiducial_xray_choice is None:
            logger.warning(f"Can't register fiducials as no X-ray is selected.")
            return

        res: tuple[list[str], torch.Tensor] | Error = self._dadg.get(
            f"{self._state.register_fiducial_xray_choice}__fiducial_points")
        if isinstance(res, Error):
            logger.warning(f"Can't find fiducial points for X-ray '{self._state.register_fiducial_xray_choice}': "
                           f"{res.description}")
            return
        xray_point_names, xray_point_vectors = res

        res: tuple[list[str], torch.Tensor] | Error = self._dadg.get("ct_fiducial_points")
        if isinstance(res, Error):
            logger.warning(f"Can't find fiducial points for CT: {res.description}")
            return
        ct_point_names, ct_point_vectors = res

        if names_not_in_both := set(xray_point_names) ^ set(ct_point_names):
            logger.warning(
                f"The following points did not have counterparts in the other image:\n{list(names_not_in_both)}")

        # rearrange ct_point_vectors so that its order corresponds to that of xray_point_vectors
        name_to_index_map = {name: index for index, name in enumerate(xray_point_names)}
        permutation = torch.tensor([name_to_index_map[name] for name in ct_point_names])
        ct_point_vectors = ct_point_vectors[permutation.argsort()]

        xray_point_vectors = np.array(xray_point_vectors)
        ct_point_vectors = np.array(ct_point_vectors)

        source_distance: float | Error = self._dadg.get(f"{self._state.register_fiducial_xray_choice}__source_distance")
        if isinstance(source_distance, Error):
            logger.warning(f"Can't find source distance: {source_distance.description}")
            return
        source = np.array([0.0, 0.0, source_distance])
        c_hat = np.array([0.0, 0.0, -1.0])

        def residuals(pose: np.ndarray) -> np.ndarray:
            homo_vectors = np.concat((ct_point_vectors, np.ones((ct_point_vectors.shape[0], 1))), axis=1)
            transformation = mapping_parameters_to_transformation(torch.tensor(pose))
            transformed_points = (homo_vectors @ np.array(transformation.get_h()))[:, 0:3]
            from_source = transformed_points - source
            frac: torch.Tensor = np.einsum("ji,i->j", from_source, c_hat) / source_distance
            projected = np.expand_dims(frac, -1) * transformed_points[:, 0:2]
            return (projected - xray_point_vectors).flatten()

        current_transformation: Transformation | Error = self._dadg.get(
            f"{self._state.register_fiducial_xray_choice}__current_transformation")
        if isinstance(current_transformation, Error):
            logger.warning(
                f"Found no current transformation for X-ray '{self._state.register_fiducial_xray_choice}': "
                f"{current_transformation.description}")
            initial_pose = np.zeros(6)
        else:
            initial_pose = np.array(mapping_transformation_to_parameters(current_transformation))

        result: sp.optimize.OptimizeResult = sp.optimize.least_squares(residuals, initial_pose, method="lm")

        if result.success:
            logger.info(f"Optimisation succeeded; x = {result.x}, loss = {result.fun}")
        else:
            logger.info(f"Optimisation failed; status = {result.status}; {result.message}")
