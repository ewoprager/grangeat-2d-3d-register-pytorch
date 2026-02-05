import argparse
import os
from typing import Any, Sequence

os.environ["QT_API"] = "PyQt6"

import torch
import napari
import pathlib

from reg23_experiments.utils import logs_setup, pushover
from reg23_experiments.io.volume import load_volume
from reg23_experiments.io.image import load_cached_drr
from reg23_experiments.data.structs import Error
from reg23_experiments.ops.data_manager import init_data_manager, data_manager, dag_updater, DAG, ChildDAG
from reg23_experiments.ops.optimisation import mapping_transformation_to_parameters, \
    mapping_parameters_to_transformation, random_parameters_at_distance
from reg23_experiments.ops import drr
from reg23_experiments.app.gui.viewer_singleton import init_viewer, viewer
from reg23_experiments.app.gui.fixed_image import FixedImageGUI
from reg23_experiments.app.gui.moving_image import MovingImageGUI
from reg23_experiments.app.gui.parameters import ParameterWidget
from reg23_experiments.experiments.parameters import Parameters, PsoParameters, NoParameters
from reg23_experiments.app.gui.register import RegisterGUI
from reg23_experiments.app.state import AppState
from reg23_experiments.app.worker_manager import WorkerManager
from reg23_experiments.data.structs import Transformation, SceneGeometry, Cropping
from reg23_experiments.ops import geometry
from reg23_experiments.ops.data_manager import updaters, args_from_dag
from reg23_experiments.ops.similarity_metric import ncc
from reg23_experiments.app.save_data_manager import SaveDataManager


@dag_updater(names_returned=["untruncated_ct_volume", "ct_spacing"])
def load_untruncated_ct(ct_path: str, device: torch.device, ct_permutation: Sequence[int] | None = None) -> dict[
    str, Any]:
    ct_volume, ct_spacing = load_volume(pathlib.Path(ct_path))
    ct_volume = ct_volume.to(device=device, dtype=torch.float32)
    ct_spacing = ct_spacing.to(device=device)

    if ct_permutation is not None:
        assert len(ct_permutation) == 3
        ct_volume = ct_volume.permute(*ct_permutation)
        ct_spacing = ct_spacing[torch.tensor(ct_permutation)]

    return {"untruncated_ct_volume": ct_volume, "ct_spacing": ct_spacing}


@dag_updater(names_returned=["source_distance", "image_2d_full", "fixed_image_spacing", "transformation_gt"])
def set_target_image(ct_path: str, ct_spacing: torch.Tensor, untruncated_ct_volume: torch.Tensor,
                     new_drr_size: torch.Size, regenerate_drr: bool, save_to_cache: bool, cache_directory: str,
                     ap_transformation: Transformation, target_ap_distance: float) -> dict[str, Any]:
    # generate a DRR through the volume
    drr_spec = None
    if not regenerate_drr:
        drr_spec = load_cached_drr(cache_directory, ct_path)

    if drr_spec is None:
        tr = mapping_parameters_to_transformation(
            random_parameters_at_distance(mapping_transformation_to_parameters(ap_transformation), target_ap_distance))
        drr_spec = drr.generate_drr_as_target(cache_directory, ct_path, untruncated_ct_volume, ct_spacing,
                                              save_to_cache=save_to_cache, size=new_drr_size, transformation=tr)

    fixed_image_spacing, scene_geometry, image_2d_full, transformation_ground_truth = drr_spec
    del drr_spec

    return {"source_distance": scene_geometry.source_distance, "image_2d_full": image_2d_full,
            "fixed_image_spacing": fixed_image_spacing, "transformation_gt": transformation_ground_truth}


@dag_updater(names_returned=["ct_volumes"])
def apply_truncation(untruncated_ct_volume: torch.Tensor, truncation_percent: int) -> dict[str, Any]:
    # truncate the volume
    truncation_fraction = 0.01 * float(truncation_percent)
    top_bottom_chop = int(round(0.5 * truncation_fraction * float(untruncated_ct_volume.size()[0])))
    ct_volume = untruncated_ct_volume[
        top_bottom_chop:max(top_bottom_chop + 1, untruncated_ct_volume.size()[0] - top_bottom_chop)]
    # mipmap the volume
    down_sampler = torch.nn.AvgPool3d(2)
    ct_volumes = [ct_volume]
    while min(ct_volumes[-1].size()) > 1:
        ct_volumes.append(down_sampler(ct_volumes[-1].unsqueeze(0))[0])
    return {"ct_volumes": ct_volumes}


@dag_updater(names_returned=["moving_image"])
def project_drr(ct_volumes: list[torch.Tensor], ct_spacing: torch.Tensor, current_transformation: Transformation,
                fixed_image_size: torch.Size, source_distance: float, fixed_image_spacing: torch.Tensor,
                downsample_level: int, translation_offset: torch.Tensor, fixed_image_offset: torch.Tensor,
                image_2d_scale_factor: float, device) -> dict[str, Any]:
    # Applying the translation offset
    new_translation = current_transformation.translation + torch.cat(
        (torch.tensor([0.0], device=device, dtype=current_transformation.translation.dtype),
         translation_offset.to(device=current_transformation.device)))
    transformation = Transformation(rotation=current_transformation.rotation, translation=new_translation).to(
        device=device)

    return {"moving_image": geometry.generate_drr(ct_volumes[downsample_level], transformation=transformation,
                                                  voxel_spacing=ct_spacing * 2.0 ** downsample_level,
                                                  detector_spacing=fixed_image_spacing / image_2d_scale_factor,
                                                  scene_geometry=SceneGeometry(source_distance=source_distance,
                                                                               fixed_image_offset=fixed_image_offset),
                                                  output_size=fixed_image_size)}


# @args_from_dag(names_left=["transformation"])
# def of(*, transformation: Transformation, ct_volumes: list[torch.Tensor], ct_spacing: torch.Tensor,
#        fixed_image_size: torch.Size, source_distance: float, fixed_image_spacing: torch.Tensor,
#        fixed_image: torch.Tensor) -> torch.Tensor:
#     moving_image = geometry.generate_drr(ct_volumes[0], transformation=transformation, voxel_spacing=ct_spacing,
#                                          detector_spacing=fixed_image_spacing,
#                                          scene_geometry=SceneGeometry(source_distance=source_distance),
#                                          output_size=fixed_image_size)
#     return ncc(moving_image, fixed_image)


def main(*, ct_path: str, cache_directory: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # -----
    # The data manager
    # -----
    init_data_manager()

    # -----
    # Viewer
    # -----
    init_viewer(title="Program Test")

    # -----
    # Updaters
    # -----
    err = data_manager().add_updater("load_untruncated_ct", load_untruncated_ct)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    err = data_manager().add_updater("apply_truncation", apply_truncation)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    err = data_manager().add_updater("set_target_image", set_target_image)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    err = data_manager().add_updater("refresh_image_2d_scale_factor", updaters.refresh_image_2d_scale_factor)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    err = data_manager().add_updater("refresh_hyperparameter_dependent", updaters.refresh_hyperparameter_dependent)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    err = data_manager().add_updater("refresh_mask_transformation_dependent",
                                     updaters.refresh_mask_transformation_dependent)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    data_manager().add_updater("project_drr", project_drr)
    if isinstance(err, Error):
        logger.error(f"Error adding updater: {err.description}")
        return

    # -----
    # Data nodes
    # -----
    data_manager().set_data_multiple(  #
        source_distance=1000.0,  #
        fixed_image_spacing=torch.tensor([0.2, 0.2]),  #
        fixed_image_size=torch.Size([500, 500]),  #
        downsample_level=0,  #
        regenerate_drr=True,  #
        cache_directory=cache_directory,  #
        save_to_cache=True,  #
        new_drr_size=torch.Size([500, 500]),  #
        ct_path=ct_path,  #
        device=device,  #
        source_offset=torch.zeros(2),  #
        mask_transformation=None,  #
        cropping=None,  #
        current_transformation=Transformation.zero(device=device),  #
        truncation_percent=0.0,  #
        ap_transformation=Transformation(rotation=torch.tensor([0.5 * torch.pi, 0.0, 0.0], device=device),
                                         translation=torch.zeros(3, device=device)),  #
        target_ap_distance=5.0,  #
    )

    # -----
    # Parameters
    # -----
    parameters = Parameters(  #
        ct_path=data_manager().get("ct_path"),  #
        downsample_level=0,  #
        truncation_percent=0,  #
        cropping="None",  #
        mask="None",  #
        sim_metric="zncc",  #
        sim_metric_parameters=NoParameters(),  #
        optimisation_algorithm="pso",  #
        op_algo_parameters=PsoParameters()  #
    )
    parameters_widget = ParameterWidget(parameters)
    viewer().window.add_dock_widget(parameters_widget, name="Params", area="right", menu=viewer().window.window_menu,
                                    tabify=True)

    app_state = AppState(parameters=parameters, dag=data_manager(),
                         transformation_save_directory=pathlib.Path("data/program_interface_transformation_save_data"))

    # -----
    # The universal objective function
    # -----
    def objective_function(dag: DAG | ChildDAG, x: torch.Tensor) -> torch.Tensor:
        t = mapping_parameters_to_transformation(x)
        # Setting the parameters
        dag.set_data("current_transformation", t)
        # Getting the resulting moving and fixed images
        moving_image = dag.get("moving_image")
        fixed_image = dag.get("fixed_image")
        # Comparing, potentially weighting with a mask
        # if apply_mask:
        #     mask = data_manager().get("mask")
        #     if weight_with_mask:
        #         return -p_sim_met.func_weighted(moving_image, fixed_image, mask)
        # return -p_sim_met.func(moving_image, fixed_image)
        return -ncc(fixed_image, moving_image)  # ToDo: configured sim metric

    # -----
    # Modules
    # -----
    worker_manager = WorkerManager(app_state=app_state, objective_function=objective_function)
    fixed_image_gui = FixedImageGUI(app_state)
    moving_image_gui = MovingImageGUI(app_state)
    register_gui = RegisterGUI(app_state)
    save_data_manager = SaveDataManager(app_state)

    value = data_manager().get("moving_image")
    if isinstance(value, Error):
        logger.error(f"Couldn't get moving image: {value.description}.")
        return

    napari.run()


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-c", "--cache-directory", type=str, default="cache",
                        help="Set the directory where data that is expensive to calculate will be saved. The default "
                             "is 'cache'.")
    parser.add_argument("-p", "--ct-path", type=str,
                        help="Give a path to a .nrrd file, .nii file or directory of .dcm files containing CT data to "
                             "process. If not "
                             "provided, some simple synthetic data will be used instead - note that in this case, "
                             "data will not be "
                             "saved to the cache.")
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
        main(ct_path=args.ct_path, cache_directory=args.cache_directory)
        if args.notify:
            pushover.send_notification(__file__, "Script finished.")
    except Exception as e:
        if args.notify:
            pushover.send_notification(__file__, "Script raised exception: {}.".format(e))
        raise e
