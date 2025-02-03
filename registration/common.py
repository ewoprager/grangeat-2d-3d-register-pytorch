from typing import NamedTuple, Tuple

import torch
from scipy.spatial.transform import Rotation
import kornia


class LinearMapping:
    def __init__(self, intercept: float | torch.Tensor, gradient: float | torch.Tensor):
        self.intercept = intercept
        self.gradient = gradient

    def __call__(self, x: float | torch.Tensor) -> float | torch.Tensor:
        return self.intercept + self.gradient * x


class LinearRange:
    def __init__(self, low: float, high: float, ):
        self.low = low
        self.high = high

    def generate_range(self, count: int, *, device) -> torch.Tensor:
        return torch.linspace(self.low, self.high, count, device=device)

    def get_mapping_from(self, other: 'LinearRange') -> LinearMapping:
        frac: float = (self.high - self.low) / (other.high - other.low)
        return LinearMapping(self.low - frac * other.low, frac)

    @classmethod
    def grid_sample_range(cls):
        return LinearRange(-1., 1.)


class Transformation(NamedTuple):
    rotation: torch.Tensor
    translation: torch.Tensor

    def inverse(self) -> 'Transformation':
        r_inverse = kornia.geometry.conversions.axis_angle_to_rotation_matrix(-self.rotation.unsqueeze(0))[0].to(
            dtype=torch.float32)
        r_inverse_t = torch.einsum('kl,...l->...k', r_inverse, self.translation.unsqueeze(0))
        return Transformation(-self.rotation, -r_inverse_t)

    @classmethod
    def zero(cls) -> 'Transformation':
        return Transformation(torch.zeros(3), torch.tensor([0., 0., 0.]))

    @classmethod
    def random(cls) -> 'Transformation':
        return Transformation(torch.pi * (-1. + 2. * torch.rand(3)),
                              25. * (-1. + 2. * torch.rand(3)) + torch.tensor([0., 0., 100.]))

    def __call__(self, positions_cartesian: torch.Tensor, exclude_translation: bool = False) -> torch.Tensor:
        device = positions_cartesian.device
        r = kornia.geometry.conversions.axis_angle_to_rotation_matrix(self.rotation[None, :])[0].to(device=device,
                                                                                                    dtype=torch.float32)
        positions_cartesian = torch.einsum('kl,...l->...k', r, positions_cartesian.to(dtype=torch.float32))
        if not exclude_translation:
            positions_cartesian = positions_cartesian + self.translation.to(device=device, dtype=torch.float32)
        return positions_cartesian


class SceneGeometry(NamedTuple):
    source_distance: float  # [mm]; distance in the positive z-direction from the centre of the detector array


class Sinogram2dGrid(NamedTuple):
    phi: torch.Tensor
    r: torch.Tensor

    def device_consistent(self) -> bool:
        return self.phi.device == self.r.device


class Sinogram3dGrid(NamedTuple):
    phi: torch.Tensor
    theta: torch.Tensor
    r: torch.Tensor

    def device_consistent(self) -> bool:
        return self.phi.device == self.theta.device and self.theta.device == self.r.device


class Sinogram2dRange(NamedTuple):
    phi: LinearRange
    r: LinearRange

    def generate_linear_grid(self, counts: int | Tuple[int] | torch.Size, *, device=torch.device("cpu")):
        if isinstance(counts, int):
            counts = (counts, counts)
        return Sinogram2dGrid(torch.linspace(self.phi.low, self.phi.high, counts[0], device=device),
                              torch.linspace(self.r.low, self.r.high, counts[1], device=device))


class Sinogram3dRange(NamedTuple):
    phi: LinearRange
    theta: LinearRange
    r: LinearRange

    def generate_linear_grid(self, counts: int | Tuple[int] | torch.Size, *, device=torch.device("cpu")):
        if isinstance(counts, int):
            counts = (counts, counts, counts)
        return Sinogram3dGrid(torch.linspace(self.phi.low, self.phi.high, counts[0], device=device),
                              torch.linspace(self.theta.low, self.theta.high, counts[1], device=device),
                              torch.linspace(self.r.low, self.r.high, counts[2], device=device))


class VolumeSpec(NamedTuple):
    ct_volume_path: str
    downsample_factor: int
    sinogram: torch.Tensor
    sinogram_range: Sinogram3dRange


class DrrSpec(NamedTuple):
    ct_volume_path: str
    detector_spacing: torch.Tensor  # [mm] distances between the detectors: (vertical, horizontal)
    scene_geometry: SceneGeometry
    image: torch.Tensor
    sinogram: torch.Tensor
    sinogram_range: Sinogram2dRange
    transformation: Transformation
