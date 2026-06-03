import logging

import torch

from reg23_app.state import AppState
from reg23_experiments.data.ct_fiducial_save_data import CTFiducialSaveManager
from reg23_experiments.data.electrode_save_data import ElectrodeSaveManager
from reg23_experiments.data.segmentation import NamedPoints2D, NamedPoints3D, OrderedPoints2D
from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.data.xray_fiducial_save_data import XRayFiducialSaveManager
from reg23_experiments.experiments import updaters
from reg23_experiments.experiments.multi_xray_truncation_updaters import project_drr, project_fiducials, \
    set_target_image
from reg23_experiments.experiments.parameters import XrayParameters
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph, NoNodeData, capture_in_namespaces
from ._gui_param_to_dag_node import cropping_changed, cropping_value_changed, respond_to_mask_change

__all__ = ["ParamDADGParityManager"]

logger = logging.getLogger(__name__)


class ParamDADGParityManager:
    """
    No GUI / widgets

    Reads from and writes to the state and DADG, and loads from save managers

    The `ParamDADGParityManager` maintains correspondence between values in the state and corresponding values in the
    DADG.
    """

    XRAY_SPECIFIC_DADG_KEYS: list[str] = ["image_2d_full", "image_2d_full_spacing", "fixed_image_spacing",
                                          "transformation_gt", "source_distance", "xray_path", "target_flipped",
                                          "moving_image", "fixed_image_size", "fixed_image_offset",
                                          "xray_sop_instance_uid", "fixed_image", "cropped_target", "mask",
                                          "translation_offset", "image_2d_scale_factor", "source_offset",
                                          "mask_transformation", "current_transformation", "cropping",
                                          "electrode_points", "fiducial_points", "projected_fiducials"]

    def __init__(self, *, state: AppState, dadg: DirectedAcyclicDataGraph, electrode_save_manager: ElectrodeSaveManager,
                 ct_fiducial_save_manager: CTFiducialSaveManager,
                 xray_fiducial_save_manager: XRayFiducialSaveManager) -> None:
        self._state = state
        self._dadg = dadg
        self._electrode_save_manager = electrode_save_manager
        self._ct_fiducial_save_manager = ct_fiducial_save_manager
        self._xray_fiducial_save_manager = xray_fiducial_save_manager

        # `ct_path` should be the same in the DADG and the state; the only necessary driving direction is state -> DADG
        self._state.parameters.observe(lambda change: self._ct_path_changed(change.new), names=["ct_path"])
        self._ct_path_changed(self._state.parameters.ct_path)

        # `downsample_level` should be the same in the DADG and the state; the only necessary driving direction is
        # state -> DADG
        self._state.parameters.observe(lambda change: self._downsample_level_changed(change.new),
                                       names=["downsample_level"])
        self._downsample_level_changed(self._state.parameters.downsample_level)

        # `truncation_percent` should be the same in the DADG and the state; the only necessary driving direction is
        # state -> DADG
        self._state.parameters.observe(lambda change: self._truncation_percent_changed(change.new),
                                       names=["truncation_percent"])
        self._truncation_percent_changed(self._state.parameters.truncation_percent)

        # mask-related nodes in the DADG should be consistent with the value of `mask` in the state; the only necessary
        # driving direction is state -> DADG
        self._state.parameters.observe(lambda change: respond_to_mask_change(dadg=self._dadg, new_value=change.new),
                                       names=["mask"])
        respond_to_mask_change(dadg=self._dadg, new_value=self._state.parameters.mask)

        # X-ray specific nodes in the DADG should be consistent with the values in `xray_parameters` in the state; the
        # only necessary driving direction is state -> DADG
        self._state.parameters.observe(lambda change: self._xray_parameters_changed(change.new),
                                       names=["xray_parameters"])
        # Need to keep track of previous set of X-rays that have been loaded
        self._current_xrays: list[str] = []
        self._xray_parameters_changed(self._state.parameters.xray_parameters)

        # load ct fiducial points from save manager
        self._dadg.observe("ct_series_uid", "fiducial_point_loader", self._ct_series_uid_changed)
        if not isinstance(ct_uid := self._dadg.get("ct_series_uid"), Error):
            self._ct_series_uid_changed(ct_uid)

        # eagerly save the ct fiducial points to the save manager
        self._dadg.observe("ct_fiducial_points", "saver", self._ct_fiducial_points_changed)

    def _ct_path_changed(self, new_value: str) -> None:
        self._dadg.set("ct_path", NoNodeData if new_value is None else new_value, check_equality=True)

    def _downsample_level_changed(self, new_value: int) -> None:
        self._dadg.set("downsample_level", new_value, check_equality=True)

    def _truncation_percent_changed(self, new_value: int) -> None:
        self._dadg.set("truncation_percent", new_value, check_equality=True)

    def _target_flipped_changed(self, new_value: bool, *, namespace: str | None) -> None:
        self._dadg.set("target_flipped" if namespace is None else f"{namespace}__target_flipped", new_value,
                       check_equality=True)

    def _xray_electrode_points_changed(self, new_value: OrderedPoints2D, *, namespace: str) -> None:
        uid: str | Error = self._dadg.get(f"{namespace}__xray_sop_instance_uid")
        if isinstance(uid, Error):
            logger.error(f"Failed to get X-ray '{namespace}' UID on electrode change: {uid.description}")
            return
        err = self._electrode_save_manager.set(  #
            uid=uid,  #
            tensor=new_value.data  #
        )
        if isinstance(err, Error):
            logger.error(f"Error saving X-ray '{namespace}' electrode point data: {err.description}")

    def _xray_fiducial_points_changed(self, new_value: NamedPoints2D, *, namespace: str) -> None:
        uid: str | Error = self._dadg.get(f"{namespace}__xray_sop_instance_uid")
        if isinstance(uid, Error):
            logger.error(f"Failed to get X-ray '{namespace}' UID on fiducial change: {uid.description}")
            return
        err = self._xray_fiducial_save_manager.set(  #
            uid=uid,  #
            names=new_value.names,  #
            points=new_value.data  #
        )
        if isinstance(err, Error):
            logger.error(f"Error saving X-ray '{namespace}' fiducial point data: {err.description}")

    def _ct_series_uid_changed(self, new_value: str) -> None:
        loaded: tuple[list[str], torch.Tensor] | None = self._ct_fiducial_save_manager.get(new_value)
        self._dadg.set("ct_fiducial_points",
                       NamedPoints3D() if loaded is None else NamedPoints3D(names=loaded[0], data=loaded[1]))

    def _ct_fiducial_points_changed(self, new_value: NamedPoints3D) -> None:
        uid: str | Error = self._dadg.get("ct_series_uid")
        if isinstance(uid, Error):
            logger.error(f"Failed to get CT UID on fiducial change: {uid.description}")
            return
        err = self._ct_fiducial_save_manager.set(  #
            uid=uid,  #
            names=new_value.names,  #
            points=new_value.data  #
        )
        if isinstance(err, Error):
            logger.error(f"Error saving CT fiducial point data: {err.description}")

    def _xray_parameters_changed(self, new_value: dict[str, XrayParameters]) -> None:
        for key, value in new_value.items():
            if key not in self._current_xrays:
                self._load_new_xray(name=key, params=value)
        self._current_xrays = list(new_value.keys())

    def _load_new_xray(self, *, name: str, params: XrayParameters):
        # Get the device
        device: torch.device | Error = self._dadg.get("device")
        if isinstance(device, Error):
            raise RuntimeError(f"Need device to load new X-ray: {device.description}")

        # Set up appropriate observers
        # for `cropping`
        params.observe(  #
            lambda change, namespace=name: cropping_changed(  #
                dadg=self._dadg,  #
                new_value=change.new,  #
                owner=change.owner,  #
                namespace=namespace  #
            ),  #
            names=["cropping"]  #
        )
        cropping_changed(dadg=self._dadg, new_value=params.cropping, owner=params, namespace=name)

        # for `cropping_value`
        params.observe(  #
            lambda change, namespace=name: cropping_value_changed(  #
                dadg=self._dadg,  #
                new_value=change.new,  #
                owner=change.owner,  #
                namespace=namespace  #
            ),  #
            names=["cropping_value"]  #
        )
        cropping_value_changed(dadg=self._dadg, new_value=params.cropping_value, owner=params, namespace=name)

        # for `target_flipped`
        params.observe(  #
            lambda change, namespace=name: self._target_flipped_changed(change.new, namespace=namespace),  #
            names=["target_flipped"]  #
        )
        self._target_flipped_changed(params.target_flipped, namespace=name)

        # for eagerly saving the `electrode_points`
        self._dadg.observe(f"{name}__electrode_points", "saver",
                           lambda new_value, _name=name: self._xray_electrode_points_changed(new_value,
                                                                                             namespace=_name))
        # for eagerly saving the `fiducial_points`
        self._dadg.observe(f"{name}__fiducial_points", "saver",
                           lambda new_value, _name=name: self._xray_fiducial_points_changed(new_value, namespace=_name))

        # Add namespaced DADG updaters
        namespace_captures = {key: name for key in ParamDADGParityManager.XRAY_SPECIFIC_DADG_KEYS}
        if isinstance(err := self._dadg.add_updater(  #
                f"{name}__refresh_image_2d_scale_factor",  #
                capture_in_namespaces(namespace_captures)(updaters.refresh_image_2d_scale_factor)), Error):
            logger.error(f"Error adding updater: {err.description}")

        if isinstance(err := self._dadg.add_updater(  #
                f"{name}__refresh_hyperparameter_dependent",  #
                capture_in_namespaces(namespace_captures)(updaters.refresh_hyperparameter_dependent)), Error):
            logger.error(f"Error adding updater: {err.description}")

        if isinstance(err := self._dadg.add_updater(  #
                f"{name}__refresh_mask_transformation_dependent",  #
                capture_in_namespaces(namespace_captures)(updaters.refresh_mask_transformation_dependent)), Error):
            logger.error(f"Error adding updater: {err.description}")

        if isinstance(err := self._dadg.add_updater(  #
                f"{name}__project_drr",  #
                capture_in_namespaces(namespace_captures)(project_drr)), Error):
            logger.error(f"Error adding updater: {err.description}")

        if isinstance(err := self._dadg.add_updater(  #
                f"{name}__set_target_image",  #
                capture_in_namespaces(namespace_captures)(set_target_image)), Error):
            logger.error(f"Error adding updater: {err.description}")

        if isinstance(err := self._dadg.add_updater(  #
                f"{name}__project_fiducials",  #
                capture_in_namespaces(namespace_captures)(project_fiducials)), Error):
            logger.error(f"Error adding updater: {err.description}")

        # Create namespaced DADG nodes
        self._dadg.set(f"{name}__xray_path", params.file_path)
        self._dadg.set(f"{name}__source_offset", torch.zeros(2, device=device))
        self._dadg.set(f"{name}__mask_transformation", None)
        self._dadg.set(f"{name}__current_transformation", Transformation.zero(device=device))
        uid: str | Error = self._dadg.get(f"{name}__xray_sop_instance_uid")
        if isinstance(uid, Error):
            logger.error(f"Error getting uid of X-ray '{name}': {uid.description}")
            return
        self._dadg.set(f"{name}__electrode_points",  #
                       OrderedPoints2D()  #
                       if (data := self._electrode_save_manager.get(uid)) is None  #
                       else OrderedPoints2D(data=data)  #
                       )
        self._dadg.set(f"{name}__fiducial_points",  #
                       NamedPoints2D()  #
                       if (res := self._xray_fiducial_save_manager.get(uid)) is None  #
                       else NamedPoints2D(names=res[0], data=res[1])  #
                       )
