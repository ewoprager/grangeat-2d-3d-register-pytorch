import logging

import pytest
import torch

from reg23_experiments.data.structs import SceneGeometry, Transformation
from reg23_experiments.ops.fiducials import *
from reg23_experiments.ops.geometry import generate_drr

logger = logging.getLogger(__name__)

_DEBUG = False


def test_fiducials() -> None:
    marker_radius = 2.5
    volume_size = torch.Size([40, 50, 60])
    spacing = torch.tensor([0.15, 0.2, 0.25], dtype=torch.float64)
    volume_size_world = spacing * (torch.tensor(volume_size, dtype=torch.float64).flip(dims=(0,)) - 1.0)
    origin = -0.5 * volume_size_world
    marker_function = lambda c: ball_bearing_marker_function(c, radius=marker_radius)
    marker_position = torch.zeros(3, dtype=torch.float64)
    volume = construct_synthetic_fiducial_3d(size=volume_size, spacing=spacing, origin=origin,
                                             marker_function=marker_function, marker_positions=[marker_position])

    guess = torch.tensor([1.0, -1.7, 2.0], dtype=torch.float64) - origin
    refined = refine_spherical_fiducial_3d(volume=volume, spacing=spacing, position=guess, radius=marker_radius)
    assert refined + origin == pytest.approx(torch.zeros(3, dtype=torch.float64))

    source_distance = 1000.0
    transformation = Transformation.zero()
    detector_spacing = torch.tensor([0.2, 0.2], dtype=torch.float64)
    detector_size = torch.Size([100, 100])
    drr_size_world = detector_spacing * (torch.tensor(detector_size, dtype=torch.float64).flip(dims=(0,)) - 1.0)
    drr = generate_drr(  #
        volume,  #
        transformation=transformation,  #
        voxel_spacing=spacing,  #
        detector_spacing=detector_spacing,  #
        scene_geometry=SceneGeometry(source_distance=source_distance),  #
        output_size=detector_size  #
    ).to(dtype=torch.float32)
    if _DEBUG:
        import matplotlib.pyplot as plt
        plt.imshow(drr)
        plt.show()
    origin_2d = 0.5 * drr_size_world
    guess_2d = torch.tensor([2.0, 3.0], dtype=torch.float64) + origin_2d
    refined_2d = refine_spherical_fiducial_2d(image=drr, spacing=detector_spacing, position=guess_2d,
                                              radius=marker_radius)
    if _DEBUG:
        logger.info(refined_2d - origin_2d)
    assert (refined_2d - origin_2d == pytest.approx(torch.zeros(2), abs=0.01))
