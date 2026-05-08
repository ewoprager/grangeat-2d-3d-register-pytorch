import logging

import numpy as np
import scipy as sp
import torch

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

        dadg_key_prefix = self._state.register_fiducial_xray_choice + "__"

        image_2d_full: torch.Tensor | Error = self._dadg.get(dadg_key_prefix + "image_2d_full")
        if isinstance(image_2d_full, Error):
            logger.error(f"Could not find 'image_2d_full' for X-ray '{self._state.register_fiducial_xray_choice}': "
                         f"{image_2d_full.description}")
            return

        fixed_image_spacing: torch.Tensor | Error = self._dadg.get(dadg_key_prefix + "fixed_image_spacing")
        if isinstance(fixed_image_spacing, Error):
            logger.error(
                f"Could not find 'fixed_image_spacing' for X-ray '{self._state.register_fiducial_xray_choice}': "
                f"{fixed_image_spacing.description}")
            return

        ct_volume: torch.Tensor | Error = self._dadg.get("untruncated_ct_volume")
        if isinstance(ct_volume, Error):
            logger.error(f"Could not find 'ct_volume': {ct_volume.description}")
            return

        ct_spacing: torch.Tensor | Error = self._dadg.get("ct_spacing")
        if isinstance(ct_spacing, Error):
            logger.error(f"Could not find 'ct_spacing': {ct_spacing.description}")
            return

        res: tuple[list[str], torch.Tensor] | Error = self._dadg.get(dadg_key_prefix + "fiducial_points")
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

        # move params to the CPU for processing
        fixed_image_spacing = fixed_image_spacing.to(device=torch.device("cpu"))
        ct_spacing = ct_spacing.to(device=torch.device("cpu"))
        xray_point_vectors = xray_point_vectors.to(device=torch.device("cpu"))
        ct_point_vectors = ct_point_vectors.to(device=torch.device("cpu"))

        if names_not_in_both := set(xray_point_names) ^ set(ct_point_names):
            logger.warning(
                f"The following points did not have counterparts in the other image:\n{list(names_not_in_both)}")

        # rearrange ct_point_vectors so that its order corresponds to that of xray_point_vectors
        name_to_index_map = {name: index for index, name in enumerate(xray_point_names)}
        permutation = torch.tensor([name_to_index_map[name] for name in ct_point_names])
        ct_point_vectors = ct_point_vectors[permutation.argsort()]

        targets_2d = xray_point_vectors - 0.5 * fixed_image_spacing * torch.tensor(image_2d_full.size()).flip(dims=(0,))
        input_points_3d = ct_point_vectors - 0.5 * ct_spacing * torch.tensor(ct_volume.size()).flip(dims=(0,))

        targets_2d = targets_2d.numpy()
        input_points_3d = input_points_3d.numpy()

        source_distance: float | Error = self._dadg.get(dadg_key_prefix + "source_distance")
        if isinstance(source_distance, Error):
            logger.warning(f"Can't find source distance: {source_distance.description}")
            return
        source = np.array([0.0, 0.0, source_distance])
        c_hat = np.array([0.0, 0.0, -1.0])

        # ToDo: switch to using the updater

        def residuals(pose: np.ndarray) -> np.ndarray:
            homo_vectors = np.concat((input_points_3d, np.ones((input_points_3d.shape[0], 1))), axis=1)
            transformation = mapping_parameters_to_transformation(torch.tensor(pose))
            transformed_points = (homo_vectors @ transformation.get_h(device=torch.device("cpu")).numpy())[:, 0:3]
            from_source = transformed_points - source
            frac = np.einsum("ji,i->j", from_source, c_hat) / source_distance
            projected = np.expand_dims(frac, -1) * transformed_points[:, 0:2]
            return (projected - targets_2d).flatten()

        current_transformation: Transformation | Error = self._dadg.get(dadg_key_prefix + "current_transformation")
        if isinstance(current_transformation, Error):
            logger.warning(f"Found no current transformation for X-ray '{self._state.register_fiducial_xray_choice}': "
                           f"{current_transformation.description}")
            initial_pose = np.zeros(6)
        else:
            initial_pose = mapping_transformation_to_parameters(current_transformation).cpu().numpy()

        result: sp.optimize.OptimizeResult = sp.optimize.least_squares(residuals, initial_pose, method="lm")

        if result.success:
            logger.info(f"Optimisation succeeded; x = {result.x}, loss = {result.fun}")
            self._dadg.set(dadg_key_prefix + "current_transformation",
                           mapping_parameters_to_transformation(torch.tensor(result.x, device=image_2d_full.device)))
        else:
            logger.info(f"Optimisation failed; status = {result.status}; {result.message}")
