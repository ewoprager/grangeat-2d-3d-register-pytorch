import argparse
import os
from typing import Literal

os.environ["QT_API"] = "PyQt6"

import torch
import napari
import pathlib

from reg23_experiments.utils import logs_setup, pushover
from reg23_experiments.data.structs import Error
from reg23_experiments.ops.data_manager import data_manager, updaters, capture_in_namespaces
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation

from reg23_experiments.app.gui.viewer_singleton import init_viewer, viewer
from reg23_experiments.app.gui.fixed_image_layer import add_fixed_image_layer
from reg23_experiments.app.gui.moving_image_layer import add_moving_image_layer
from reg23_experiments.app.gui.electrode_layer import add_electrode_layer
from reg23_experiments.app.gui.parameters import ParametersWidget
from reg23_experiments.app.gui.helpers import TraitletsWidget
from reg23_experiments.experiments.parameters import Parameters, PsoParameters, NoParameters, Context
from reg23_experiments.app.gui.register_widget import RegisterWidget
from reg23_experiments.app.context import AppContext
from reg23_experiments.app.worker_manager import WorkerManager
from reg23_experiments.data.structs import Transformation
from reg23_experiments.ops.similarity_metric import ncc
from reg23_experiments.app.transformation_saver import TransformationSaver
from reg23_experiments.app.gui.images_widget import ImagesWidget
from reg23_experiments.experiments.multi_xray_truncation_updaters import load_untruncated_ct, set_target_image, \
    apply_truncation, project_drr, read_xray_uid


# @args_from_dag(names_left=["transformation"])
# def of(*, transformation: Transformation, ct_volumes: list[torch.Tensor], ct_spacing: torch.Tensor,
#        fixed_image_size: torch.Size, source_distance: float, fixed_image_spacing: torch.Tensor,
#        fixed_image: torch.Tensor) -> torch.Tensor:
#     moving_image = geometry.generate_drr(ct_volumes[0], transformation=transformation, voxel_spacing=ct_spacing,
#                                          detector_spacing=fixed_image_spacing,
#                                          scene_geometry=SceneGeometry(source_distance=source_distance),
#                                          output_size=fixed_image_size)
#     return ncc(moving_image, fixed_image)


def main(*, ct_path: str | None = None, xray_path: str | None = None,
         external_dataset: Literal["gold_hip"] | None = None, cache_directory: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # -----
    # Viewer
    # -----
    init_viewer(title="Program Test")

    # -----
    # Updaters
    # -----
    err = data_manager().add_updater("apply_truncation", apply_truncation)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return

    # -----
    # Data nodes
    # -----
    data_manager().set_multiple(  #
        downsample_level=0,  #
        regenerate_drr=True,  #
        cache_directory=cache_directory,  #
        save_to_cache=True,  # a
        new_drr_size=torch.Size([500, 500]),  #
        device=device,  #
        cropping=None,  #
        current_transformation=Transformation.zero(device=device),  #
        truncation_percent=0.0,  #
        ap_transformation=Transformation(rotation=torch.tensor([0.5 * torch.pi, 0.0, 0.0], device=device),
                                         translation=torch.zeros(3, device=device)),  #
        target_ap_distance=5.0,  #
    )
    if ct_path is not None:
        data_manager().set("ct_path", ct_path)

    # -----
    # External datasets
    # -----
    if external_dataset == "gold_hip":
        from reg23_experiments.io.external_datasets import gold_hip
        # CT data
        untruncated_ct_volume, ct_spacing = gold_hip.load_ct()
        untruncated_ct_volume = untruncated_ct_volume.to(device=data_manager().get("device"))
        ct_spacing = ct_spacing.to(device=data_manager().get("device"))
        data_manager().set_multiple(  #
            untruncated_ct_volume=untruncated_ct_volume,  #
            ct_spacing=ct_spacing,  #
            ct_path=gold_hip.get_data_config().get_ct_path()  #
        )
        # X-ray data
        xray_data = gold_hip.load_xray("p19", untruncated_ct_volume.size(), ct_spacing)
        logger.info(f"ct spacing = {ct_spacing.cpu()}")
        logger.info(f"ct size = {untruncated_ct_volume.size()}")
        logger.info(f"total = "
                    f"{ct_spacing.cpu() * torch.tensor(untruncated_ct_volume.size(), dtype=torch.float64).flip(dims=(0,))}")
        image_2d_full = xray_data["image"].to(device=data_manager().get("device"))
        data_manager().set_multiple(  #
            source_distance=xray_data["scene_geometry"].source_distance,  #
            image_2d_full=image_2d_full,  #
            fixed_image_spacing=xray_data["spacing"],  #
            source_offset=xray_data["scene_geometry"].fixed_image_offset,  #
            transformation_gt=xray_data["transformation"].to(device=data_manager().get("device"))  #
        )

    else:
        err = data_manager().add_updater("load_untruncated_ct", load_untruncated_ct)
        if isinstance(err, Error):
            logger.error(f"Error adding updater: {err.description}")
            return

    # -----
    # Parameters
    # -----
    parameters = Parameters(  #
        ct_path=None,  #
        downsample_level=0,  #
        truncation_percent=0,  #
        mask="None",  #
        sim_metric="zncc",  #
        sim_metric_parameters=NoParameters(),  #
        starting_distance=0.0,  #
        sample_count_per_distance=1,  #
        optimisation_algorithm="pso",  #
        op_algo_parameters=PsoParameters(),  #
        iteration_count=10,  #
    )

    # from reg23_experiments.utils.data import clone_has_traits
    # parameters.op_algo_parameters.particle_count = 5
    # test = clone_has_traits(parameters)

    app_context = AppContext(parameters=parameters, dadg=data_manager(),
                             transformation_save_directory=pathlib.Path("data/app_transformation_save_data"),
                             electrode_save_directory=pathlib.Path("data/app_electrode_save_data"))

    parameters_widget = ParametersWidget(app_context)
    viewer().window.add_dock_widget(parameters_widget, name="Params", area="right", menu=viewer().window.window_menu,
                                    tabify=True)
    viewer().window.add_dock_widget(TraitletsWidget(app_context.state.gui_settings), name="GUI Settings", area="left",
                                    menu=viewer().window.window_menu)

    images_widget = ImagesWidget(app_context)
    viewer().window.add_dock_widget(images_widget, name="Images", area="right", menu=viewer().window.window_menu,
                                    tabify=True)

    # -----
    # The universal objective function
    # -----
    def objective_function(context: Context, x: torch.Tensor) -> torch.Tensor:
        t = mapping_parameters_to_transformation(x)
        # Setting the parameters
        context.dadg.set(
            "current_transformation" if context.namespace is None else f"{context.namespace}__current_transformation",
            t)
        # Getting the resulting moving and fixed images
        moving_image = context.dadg.get(
            "moving_image" if context.namespace is None else f"{context.namespace}__moving_image")
        fixed_image = context.dadg.get(
            "fixed_image" if context.namespace is None else f"{context.namespace}__fixed_image")
        # Comparing, potentially weighting with a mask
        # if apply_mask:
        #     mask = data_manager().get("mask")
        #     if weight_with_mask:
        #         return -p_sim_met.func_weighted(moving_image, fixed_image, mask)
        # return -p_sim_met.func(moving_image, fixed_image)
        return -ncc(fixed_image, moving_image)  # ToDo: configured sim metric

    # -----
    # GUI Modules
    # -----
    register_widget = RegisterWidget(app_context)

    # -----
    # Modules
    # -----
    worker_manager = WorkerManager(ctx=app_context, objective_function=objective_function)
    transformation_saver = TransformationSaver(app_context)

    # value = data_manager().get("a__moving_image")
    # if isinstance(value, Error):
    #     logger.error(f"Couldn't get moving image: {value.description}.")
    #     return

    napari.run()


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-c", "--cache-directory", type=str, default="cache",
                        help="Set the directory where data that is expensive to calculate will be saved. The default "
                             "is 'cache'.")
    parser.add_argument("-p", "--ct-path", type=str, default=None,
                        help="Give a path to a .nrrd file, .nii file or directory of .dcm files containing CT data to "
                             "process.")
    parser.add_argument("--external-gold-hip", action="store_true",
                        help="Load data from the external dataset '2D-3D registration gold-standard dataset for the "
                             "hip joint based on uncertainty modeling'")
    parser.add_argument("-x", "--xray-path", type=str, default=None,
                        help="Give a path to a DICOM file containing an X-ray image to register the CT image to. If "
                             "this is provided, the X-ray will by used instead of any DRR.")
    # parser.add_argument("-i", "--no-load", action='store_true',
    #                     help="Do not load any pre-calculated data from the cache.")
    # parser.add_argument(
    #     "-r", "--regenerate-drr", action='store_true',
    #     help="Regenerate the DRR through the 3D data, regardless of whether a DRR has been cached.")
    # parser.add_argument("-n", "--no-save", action='store_true', help="Do not save any data to the cache.")
    parser.add_argument("-n", "--notify", action="store_true", help="Send notification on completion.")
    # parser.add_argument("-s", "--show", action="store_true", help="Show images at the G.T. alignment.")
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    try:
        if args.external_gold_hip:
            main(external_dataset="gold_hip", cache_directory=args.cache_directory)
        else:
            main(ct_path=args.ct_path, xray_path=args.xray_path, cache_directory=args.cache_directory)
        if args.notify:
            pushover.send_notification(__file__, "Script finished.")
    except Exception as e:
        if args.notify:
            pushover.send_notification(__file__, "Script raised exception: {}.".format(e))
        raise e
