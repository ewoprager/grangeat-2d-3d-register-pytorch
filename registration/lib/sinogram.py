import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

import Extension

from registration.lib.structs import *
from registration.lib import geometry
from registration.lib import grangeat


class Sinogram(ABC):
    @abstractmethod
    def to(self, **kwargs) -> 'Sinogram':
        pass

    @abstractmethod
    def device(self):
        pass

    @abstractmethod
    def resample(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid) -> torch.Tensor:
        pass

    @abstractmethod
    def resample_python(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid) -> torch.Tensor:
        pass


class SinogramClassic(Sinogram):
    def __init__(self, data: torch.Tensor, sinogram_range: Sinogram3dRange):
        self.data = data
        self.sinogram_range = sinogram_range

    def to(self, **kwargs) -> 'SinogramClassic':
        return SinogramClassic(self.data.to(**kwargs), self.sinogram_range)

    def device(self):
        return self.data.device

    def get_spacing(self, *, device=torch.device('cpu')) -> torch.Tensor:
        return self.sinogram_range.get_spacing(self.data.size(), device=device)

    def resample(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid) -> torch.Tensor:
        device = self.data.device
        sinogram_range_low = torch.tensor(
            [self.sinogram_range.r.low, self.sinogram_range.theta.low, self.sinogram_range.phi.low], device=device)
        sinogram_range_high = torch.tensor(
            [self.sinogram_range.r.high, self.sinogram_range.theta.high, self.sinogram_range.phi.high], device=device)
        sinogram_spacing = (sinogram_range_high - sinogram_range_low) / (
            torch.tensor(self.data.size(), dtype=torch.float32, device=device).flip(dims=(0,)))
        sinogram_range_centres = .5 * (sinogram_range_low + sinogram_range_high)
        return Extension.resample_sinogram3d(self.data, sinogram_spacing, sinogram_range_centres, ph_matrix,
                                             fixed_image_grid.phi, fixed_image_grid.r)

    def resample_python(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid,
                        smooth: bool = False) -> torch.Tensor:
        assert fixed_image_grid.device_consistent()
        assert fixed_image_grid.phi.device == self.device()
        assert ph_matrix.device == self.device()

        fixed_image_grid_cartesian = geometry.fixed_polar_to_moving_cartesian(fixed_image_grid, ph_matrix=ph_matrix)

        fixed_image_grid_sph = geometry.moving_cartesian_to_moving_spherical(fixed_image_grid_cartesian)

        ## sign changes - this relies on the convenient coordinate system
        moving_origin_projected = ph_matrix[0:2, 3] / ph_matrix[3, 3]
        square_radius: torch.Tensor = .25 * moving_origin_projected.square().sum()
        need_sign_change = ((fixed_image_grid.r.unsqueeze(-1) * torch.stack(
            (torch.cos(fixed_image_grid.phi), torch.sin(fixed_image_grid.phi)),
            dim=-1) - .5 * moving_origin_projected).square().sum(dim=-1) < square_radius)
        ##

        ##
        # _, axes = plt.subplots()
        # mesh = axes.pcolormesh(fixed_image_grid_sph.phi.cpu())
        # axes.axis('square')
        # axes.set_title("phi_sph resampling values")
        # axes.set_xlabel("r_pol")
        # axes.set_ylabel("phi_pol")
        # plt.colorbar(mesh)
        # _, axes = plt.subplots()
        # mesh = axes.pcolormesh(fixed_image_grid_sph.theta.cpu())
        # axes.axis('square')
        # axes.set_title("theta_sph resampling values")
        # axes.set_xlabel("r_pol")
        # axes.set_ylabel("phi_pol")
        # plt.colorbar(mesh)
        # _, axes = plt.subplots()
        # mesh = axes.pcolormesh(fixed_image_grid_sph.r.cpu())
        # axes.axis('square')
        # axes.set_title("r_sph resampling values")
        # axes.set_xlabel("r_pol")
        # axes.set_ylabel("phi_pol")
        # plt.colorbar(mesh)
        ##

        grid_range = LinearRange.grid_sample_range()
        i_mapping: LinearMapping = grid_range.get_mapping_from(self.sinogram_range.r)
        j_mapping: LinearMapping = grid_range.get_mapping_from(self.sinogram_range.theta)
        k_mapping: LinearMapping = grid_range.get_mapping_from(self.sinogram_range.phi)

        if smooth:
            ret = grangeat.grid_sample_sinogram3d_smoothed(self.data, fixed_image_grid_sph.phi,
                                                           fixed_image_grid_sph.theta, fixed_image_grid_sph.r,
                                                           i_mapping=i_mapping, j_mapping=j_mapping,
                                                           k_mapping=k_mapping)
        else:
            grid = torch.stack((i_mapping(fixed_image_grid_sph.r), j_mapping(fixed_image_grid_sph.theta),
                                k_mapping(fixed_image_grid_sph.phi)), dim=-1)
            ret = torch.nn.functional.grid_sample(self.data[None, None, :, :, :], grid[None, None, :, :, :])[0, 0, 0]

        ret[need_sign_change] *= -1.
        return ret


class SinogramFibonacci(Sinogram):
    def __init__(self, data: torch.Tensor, r_range: LinearRange):
        self.data = data
        self.r_range = r_range

    def to(self, **kwargs) -> 'SinogramFibonacci':
        return SinogramFibonacci(self.data.to(**kwargs), self.r_range)

    def device(self):
        return self.data.device

    def resample(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid) -> torch.Tensor:
        raise Exception("Not yet implemented")

    def resample_python(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid) -> torch.Tensor:
        raise Exception("Not yet implemented")


class VolumeSpec(NamedTuple):
    ct_volume_path: str
    downsample_factor: int
    sinogram: SinogramClassic


class VolumeSpecFibonacci(NamedTuple):
    ct_volume_path: str
    downsample_factor: int
    sinogram: SinogramFibonacci


class DrrSpec(NamedTuple):
    ct_volume_path: str
    detector_spacing: torch.Tensor  # [mm] distances between the detectors: (vertical, horizontal)
    scene_geometry: SceneGeometry
    image: torch.Tensor
    sinogram: torch.Tensor
    sinogram_range: Sinogram2dRange
    transformation: Transformation
