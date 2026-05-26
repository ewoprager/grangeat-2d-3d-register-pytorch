import logging

import pytest
import torch

from reg23_experiments.ops.fiducials import *

logger = logging.getLogger(__name__)


def test_fiducials_3d() -> None:
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
