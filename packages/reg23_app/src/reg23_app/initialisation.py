import torch
import traitlets
from reg23_app._gui_param_to_dag_node import cropping_changed, cropping_value_changed, respond_to_mask_change
from reg23_app.context import AppContext
from reg23_app.param_dadg_parity_manager import ParamDADGParityManager

from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.experiments import updaters
from reg23_experiments.experiments.multi_xray_truncation_updaters import project_drr
from reg23_experiments.experiments.parameters import XrayParameters
from reg23_experiments.ops.data_manager import capture_in_namespaces

__all__ = ["FixedTarget", "load_fixed_dataset"]


class FixedTarget(traitlets.HasTraits):
    source_distance: float = traitlets.Float()
    image_2d_full: torch.Tensor = traitlets.Instance(torch.Tensor, allow_none=False)
    fixed_image_spacing: torch.Tensor = traitlets.Instance(torch.Tensor, allow_none=False)
    transformation_gt: Transformation = traitlets.Instance(Transformation, allow_none=False)


def load_fixed_dataset(*,  #
                       app_context: AppContext,  #
                       ct_volume: torch.Tensor,  #
                       ct_spacing: torch.Tensor,  #
                       fixed_targets: dict[str, FixedTarget]) -> None | Error:
    device = app_context.dadg.get("device")
    # CT volume
    ct_volume = ct_volume.to(device=device)
    ct_spacing = ct_spacing.to(device=device)
    res = app_context.dadg.set_multiple(  #
        untruncated_ct_volume=ct_volume,  #
        ct_spacing=ct_spacing,  #
        ct_path="<fixed_dataset>"  #
    )
    if isinstance(res, Error):
        return Error(f"Failed to set gold_hip CT dadg nodes: {res.description}")
    # X-ray images
    for name, fixed_target in fixed_targets.items():
        # Add X-ray params
        params = XrayParameters(xray_path="<fixed_dataset>")
        app_context.state.parameters.xray_parameters[name] = params

        # Set up appropriate observers
        # for `cropping`
        params.observe(  #
            lambda change, namespace=name: cropping_changed(  #
                dadg=app_context.dadg,  #
                new_value=change.new,  #
                owner=change.owner,  #
                namespace=namespace  #
            ),  #
            names=["cropping"]  #
        )
        cropping_changed(dadg=app_context.dadg, new_value=params.cropping, owner=params, namespace=name)

        # for `cropping_value`
        params.observe(  #
            lambda change, namespace=name: cropping_value_changed(  #
                dadg=app_context.dadg,  #
                new_value=change.new,  #
                owner=change.owner,  #
                namespace=namespace  #
            ),  #
            names=["cropping_value"]  #
        )
        cropping_value_changed(dadg=app_context.dadg, new_value=params.cropping_value, owner=params, namespace=name)

        # for `target_flipped`
        # ToDo: this doesn't work

        # Create namespaced DADG nodes
        app_context.dadg.set(f"{name}__source_distance", fixed_target.source_distance)
        app_context.dadg.set(f"{name}__image_2d_full", fixed_target.image_2d_full.to(device=device))
        app_context.dadg.set(f"{name}__fixed_image_spacing", fixed_target.fixed_image_spacing)
        app_context.dadg.set(f"{name}__transformation_gt", fixed_target.transformation_gt.to(device=device))
        app_context.dadg.set(f"{name}__xray_sop_instance_uid", f"fixed_target_{name}")
        app_context.dadg.set(f"{name}__xray_path", params.file_path)
        app_context.dadg.set(f"{name}__source_offset", torch.zeros(2))
        app_context.dadg.set(f"{name}__mask_transformation", None)
        app_context.dadg.set(f"{name}__current_transformation", Transformation.zero(device=device))
        # Add namespaced DADG updaters
        namespace_captures = {key: name for key in ParamDADGParityManager.XRAY_SPECIFIC_DADG_KEYS}
        err = app_context.dadg.add_updater(f"{name}__refresh_image_2d_scale_factor",
                                           capture_in_namespaces(namespace_captures)(
                                               updaters.refresh_image_2d_scale_factor))
        if isinstance(err, Error):
            return Error(f"Error adding updater: {err.description}")

        err = app_context.dadg.add_updater(f"{name}__refresh_hyperparameter_dependent",
                                           capture_in_namespaces(namespace_captures)(
                                               updaters.refresh_hyperparameter_dependent))
        if isinstance(err, Error):
            return Error(f"Error adding updater: {err.description}")

        err = app_context.dadg.add_updater(f"{name}__refresh_mask_transformation_dependent",
                                           capture_in_namespaces(namespace_captures)(
                                               updaters.refresh_mask_transformation_dependent))
        if isinstance(err, Error):
            return Error(f"Error adding updater: {err.description}")

        err = app_context.dadg.add_updater(f"{name}__project_drr",
                                           capture_in_namespaces(namespace_captures)(project_drr))
        if isinstance(err, Error):
            return Error(f"Error adding updater: {err.description}")

    return None
