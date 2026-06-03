import logging

import numpy as np
import scipy as sp
import torch

from reg23_app.state import AppState
from reg23_experiments.data.segmentation import NamedPoints2D, NamedPoints3D
from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph
from reg23_experiments.ops.fiducials import refine_spherical_fiducial_2d, refine_spherical_fiducial_3d
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
        self._state.observe(self._button_ct_fiducial_refine, names=["button_refine_ct_fiducials"])
        self._state.observe(self._button_xray_fiducial_refine, names=["button_refine_xray_fiducials"])

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

        xray_fiducial_points: NamedPoints2D | Error = self._dadg.get(dadg_key_prefix + "fiducial_points")
        if isinstance(xray_fiducial_points, Error):
            logger.warning(f"Can't find fiducial points for X-ray '{self._state.register_fiducial_xray_choice}': "
                           f"{xray_fiducial_points.description}")
            return

        if len(xray_fiducial_points.names) < 3:
            logger.warning(f"Cannot register using fiducials with fewer than 3 markers.")
            return

        device: torch.device | Error = self._dadg.get("device")
        if isinstance(device, Error):
            raise Exception("Failed to get device from DADG")

        # ToDo: Speed this up
        def residuals(pose: np.ndarray) -> np.ndarray:
            self._dadg.set(dadg_key_prefix + "current_transformation",
                           mapping_parameters_to_transformation(torch.tensor(pose, device=device)))
            projected_points: NamedPoints2D | Error = self._dadg.get(dadg_key_prefix + "projected_fiducials")
            if isinstance(projected_points, Error):
                raise Exception(
                    f"Failed to get projected fiducials during optimisation: {projected_points.description}")
            projected_name_to_index = {name: index for index, name in enumerate(projected_points.names)}
            ordered = projected_points.data[
                torch.tensor([projected_name_to_index[name] for name in xray_fiducial_points.names])]
            return (ordered - xray_fiducial_points.data).flatten().numpy()

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

    def _button_ct_fiducial_refine(self, change) -> None:
        if not change.new:
            return
        self._state.button_refine_ct_fiducials = False

        logger.info(f"Refining fiducial segmentation for CT")

        untruncated_ct_volume: torch.Tensor | Error = self._dadg.get("untruncated_ct_volume")
        if isinstance(untruncated_ct_volume, Error):
            logger.error(
                f"Couldn't get 'untruncated_ct_volume' for fiducial refinement: {untruncated_ct_volume.description}")
            return

        ct_spacing: torch.Tensor | Error = self._dadg.get("ct_spacing")
        if isinstance(ct_spacing, Error):
            logger.error(f"Couldn't get 'ct_spacing' for fiducial refinement: {ct_spacing.description}")
            return

        fiducial_points: NamedPoints3D | Error = self._dadg.get("ct_fiducial_points")
        if isinstance(fiducial_points, Error):
            logger.error(f"Couldn't get 'ct_fiducial_points' for fiducial refinement: {fiducial_points.description}")
            return

        new_data = torch.empty_like(fiducial_points.data)
        for i, point in enumerate(fiducial_points.data):
            new_data[i] = refine_spherical_fiducial_3d(  #
                volume=untruncated_ct_volume,  #
                spacing=ct_spacing,  #
                position=point,  #
                radius=0.5 * self._state.assumed_fiducial_diameter,  #
            )
        self._dadg.set("ct_fiducial_points", NamedPoints3D(names=fiducial_points.names, data=new_data))

        logger.info("CT fiducial segmentation refined.")

    def _button_xray_fiducial_refine(self, change) -> None:
        if not change.new:
            return
        self._state.button_refine_xray_fiducials = False

        logger.info(f"Refining fiducial segmentation for X-ray '{self._state.register_fiducial_xray_choice}'")

        dadg_key_prefix = self._state.register_fiducial_xray_choice + "__"

        image_2d_full: torch.Tensor | Error = self._dadg.get(dadg_key_prefix + "image_2d_full")
        if isinstance(image_2d_full, Error):
            logger.error(
                f"Couldn't get '{dadg_key_prefix}image_2d_full' for fiducial refinement: {image_2d_full.description}")
            return

        image_2d_full_spacing: torch.Tensor | Error = self._dadg.get(dadg_key_prefix + "image_2d_full_spacing")
        if isinstance(image_2d_full_spacing, Error):
            logger.error(f"Couldn't get '{dadg_key_prefix}image_2d_full_spacing' for fiducial refinement: "
                         f"{image_2d_full_spacing.description}")
            return

        fiducial_points: NamedPoints2D | Error = self._dadg.get(dadg_key_prefix + "fiducial_points")
        if isinstance(fiducial_points, Error):
            logger.warning(f"Can't find fiducial points for X-ray '{self._state.register_fiducial_xray_choice}': "
                           f"{fiducial_points.description}")
            return

        new_data = torch.empty_like(fiducial_points.data)
        for i, point in enumerate(fiducial_points.data):
            new_data[i] = refine_spherical_fiducial_2d(  #
                image=image_2d_full,  #
                spacing=image_2d_full_spacing,  #
                position=point,  #
                radius=0.5 * self._state.assumed_fiducial_diameter,  #
            )
        self._dadg.set(dadg_key_prefix + "fiducial_points", NamedPoints2D(names=fiducial_points.names, data=new_data))

        logger.info("X-ray fiducial segmentation refined.")
