import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

import pyvista as pv

import Extension

from registration.lib.structs import *
from registration.lib import geometry
from registration.lib import plot as myplt


class Sinogram(ABC):
    @abstractmethod
    def to(self, **kwargs) -> 'Sinogram':
        pass

    @property
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
    phi_range = LinearRange(-0.5 * torch.pi, 0.5 * torch.pi)

    @staticmethod
    def theta_range_from_count(theta_count: int) -> LinearRange:
        return LinearRange(-0.5 * torch.pi, torch.pi * (.5 - 1. / float(theta_count)))

    @staticmethod
    def build_grid(*, counts: int | Tuple[int, int, int] | torch.Size, r_range: LinearRange,
                   device=torch.device("cpu")) -> Sinogram3dGrid:
        if isinstance(counts, int):
            counts = (counts, counts, counts)
        elif isinstance(counts, torch.Size):
            assert len(counts) == 3
        phis = SinogramClassic.phi_range.generate_grid(counts[0], device=device)
        thetas = SinogramClassic.theta_range_from_count(counts[1]).generate_grid(counts[1], device=device)
        rs = r_range.generate_grid(counts[2], device=device)
        phis, thetas, rs = torch.meshgrid(phis, thetas, rs)
        return Sinogram3dGrid(phis, thetas, rs)

    def __init__(self, data: torch.Tensor, r_range: LinearRange):
        assert len(data.size()) == 3
        self.data = data
        self.r_range = r_range

    @property
    def device(self):
        return self.data.device

    @property
    def r_range(self) -> LinearRange:
        return self.r_range

    @property
    def theta_range(self) -> LinearRange:
        theta_count: int = self.data.size()[1]
        return SinogramClassic.theta_range_from_count(theta_count)

    @property
    def grid(self) -> Sinogram3dGrid:
        return SinogramClassic.build_grid(counts=self.data.size(), r_range=self.r_range, device=self.device)

    def to(self, **kwargs) -> 'SinogramClassic':
        return SinogramClassic(self.data.to(**kwargs), self.r_range)

    def resample(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid) -> torch.Tensor:
        return Extension.resample_sinogram3d(
            self.data, "classic", self.r_range.get_spacing(self.data.size()[2]), ph_matrix, fixed_image_grid.phi,
            fixed_image_grid.r)

    def resample_python(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid, *, smooth: float | None = None,
                        plot: bool = False) -> torch.Tensor:
        assert fixed_image_grid.device_consistent()
        assert fixed_image_grid.phi.device == self.device
        assert ph_matrix.device == self.device

        fixed_image_grid_sph = geometry.fixed_polar_to_moving_spherical(
            fixed_image_grid, ph_matrix=ph_matrix, plot=plot)

        if plot:
            myplt.visualise_planes_as_points(fixed_image_grid_sph, fixed_image_grid_sph.phi)
            _, axes = plt.subplots()
            mesh = axes.pcolormesh(fixed_image_grid_sph.phi.cpu())
            axes.axis('square')
            axes.set_title("phi_sph resampling values")
            axes.set_xlabel("r_pol")
            axes.set_ylabel("phi_pol")
            plt.colorbar(mesh)
            _, axes = plt.subplots()
            mesh = axes.pcolormesh(fixed_image_grid_sph.theta.cpu())
            axes.axis('square')
            axes.set_title("theta_sph resampling values")
            axes.set_xlabel("r_pol")
            axes.set_ylabel("phi_pol")
            plt.colorbar(mesh)
            _, axes = plt.subplots()
            mesh = axes.pcolormesh(fixed_image_grid_sph.r.cpu())
            axes.axis('square')
            axes.set_title("r_sph resampling values")
            axes.set_xlabel("r_pol")
            axes.set_ylabel("phi_pol")
            plt.colorbar(mesh)

        grid_range = LinearRange.grid_sample_range()
        i_mapping: LinearMapping = grid_range.get_mapping_from(self.r_range)
        j_mapping: LinearMapping = grid_range.get_mapping_from(self.theta_range)
        k_mapping: LinearMapping = grid_range.get_mapping_from(self.phi_range)

        if smooth is not None:
            ret = self.grid_sample_smoothed(
                fixed_image_grid_sph, i_mapping=i_mapping, j_mapping=j_mapping, k_mapping=k_mapping, sigma=smooth)
        else:
            grid = torch.stack(
                (
                    i_mapping(fixed_image_grid_sph.r), j_mapping(fixed_image_grid_sph.theta),
                    k_mapping(fixed_image_grid_sph.phi)), dim=-1)
            ret = Extension.grid_sample3d(self.data, grid, "wrap")

        del fixed_image_grid_sph

        ## sign changes - this implementation relies on the convenient coordinate system
        moving_origin_projected = ph_matrix[0:2, 3] / ph_matrix[3, 3]
        square_radius: torch.Tensor = .25 * moving_origin_projected.square().sum()
        need_sign_change = ((fixed_image_grid.r.unsqueeze(-1) * torch.stack(
            (torch.cos(fixed_image_grid.phi), torch.sin(fixed_image_grid.phi)),
            dim=-1) - .5 * moving_origin_projected).square().sum(dim=-1) < square_radius)
        ##

        ret[need_sign_change] *= -1.

        del need_sign_change

        return ret

    def grid_sample_smoothed(self, grid: Sinogram3dGrid, *, i_mapping: LinearMapping, j_mapping: LinearMapping,
                             k_mapping: LinearMapping, sigma: float, offset_count: int = 10):
        """
        Sample the sinogram at the given phi, theta, r spherical coordinates, with extra samples in a Gaussian layout
        around the sampling positions to make the sampling more even over S^2, even if the point distribution is less
        even.

        :param grid: A grid of 3D sinogram coordinates
        :param i_mapping: Mapping from r to sinogram texture x-coordinate (-1, 1)
        :param j_mapping: Mapping from theta to sinogram texture y-coordinate (-1, 1)
        :param k_mapping: Mapping from phi to sinogram texture z-coordinate (-1, 1)
        :param offset_count: Number of rows and columns of offset points to make weighted samples at
        :param sigma: The standard deviation of the Gaussian pattern
        :return: A tensor matching size of `phi_values`  - the weighted sums of offset samples around the given
        coordinates.
        """

        assert grid.phi.device == self.device
        assert grid.device_consistent()
        assert grid.size_consistent()
        assert sigma >= 0.

        logger.info("Sample smoothing with sigma = {:.3f}".format(sigma))

        # Gaussian weighed sampling offsets
        xs = torch.linspace(-3. * sigma, 3. * sigma, offset_count)
        ys = torch.linspace(-3. * sigma, 3. * sigma, offset_count)
        ys, xs = torch.meshgrid(ys, xs)
        xs = xs.flatten()
        ys = ys.flatten()
        weights = (-(xs.square() + ys.square()) / (2. * sigma * sigma)).exp()
        weights = weights / weights.sum()

        # New offset values of phi & theta are determined by rotating the vector (1, 0, 0)^T first by a small
        # perturbation
        # according to the Gaussian pattern, and then by the original rotation according to the old values of phi *
        # theta.

        # Determining a perturbed vector for each offset in the Gaussian pattern:
        phis_off = torch.atan2(ys, xs)
        thetas_off = (xs.square() + ys.square()).sqrt()

        del xs, ys

        cp = phis_off.cos()
        sp = phis_off.sin()
        ct = thetas_off.cos()
        st = thetas_off.sin()

        del phis_off, thetas_off

        offset_vectors = torch.stack((ct, -st * sp, st * cp), dim=-1).to(device=self.device)

        # Determining the rotation matrices for each input (phi, theta):
        cp = grid.phi.cos()
        sp = grid.phi.sin()
        ct = grid.theta.cos()
        st = grid.theta.sin()
        row_0 = torch.stack((cp * ct, -sp, -cp * st), dim=-1)
        row_1 = torch.stack((sp * ct, cp, sp * st), dim=-1)
        row_2 = torch.stack((st, torch.zeros_like(st), ct), dim=-1)
        rotation_matrices = torch.stack((row_0, row_1, row_2), dim=-2).to(device=self.device)
        # Multiplying by the perturbed unit vectors for the perturbed, rotated unit vector:
        rotated_vectors = torch.einsum('...ij,kj->...ki', rotation_matrices, offset_vectors)

        del cp, sp, ct, st, row_0, row_1, row_2, rotation_matrices, offset_vectors

        # Converting the resulting unit vectors back into new values of phi & theta, and expanding the r tensor to match
        # in size:
        new_phis = torch.atan2(rotated_vectors[..., 1], rotated_vectors[..., 0])
        new_thetas = torch.clamp(rotated_vectors[..., 2], -1., 1.).asin()

        del rotated_vectors

        new_grid = Sinogram3dGrid(
            new_phis, new_thetas, grid.r.unsqueeze(-1).expand(
                [-1] * len(grid.r.size()) + [new_phis.size()[-1]]).clone()).unflip()  # need to
        # clone the r grid otherwise it's just a tensor view, not a tensor in its own right

        del new_phis, new_thetas

        # Sampling at all the perturbed orientations:
        grid = torch.stack((i_mapping(new_grid.r), j_mapping(new_grid.theta), k_mapping(new_grid.phi)), dim=-1)
        samples = Extension.grid_sample3d(self.data, grid, address_mode="wrap")

        del grid

        ##
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(torch.einsum('i,...i->...', weights, new_grid.phi.cpu()).cpu())
        axes.axis('square')
        axes.set_title("average phi_sph resampling values")
        axes.set_xlabel("r_pol")
        axes.set_ylabel("phi_pol")
        plt.colorbar(mesh)
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(torch.einsum('i,...i->...', weights, new_grid.theta.cpu()).cpu())
        axes.axis('square')
        axes.set_title("average theta_sph resampling values")
        axes.set_xlabel("r_pol")
        axes.set_ylabel("phi_pol")
        plt.colorbar(mesh)
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(torch.einsum('i,...i->...', weights, new_grid.r.cpu()).cpu())
        axes.axis('square')
        axes.set_title("average r_sph resampling values")
        axes.set_xlabel("r_pol")
        axes.set_ylabel("phi_pol")
        plt.colorbar(mesh)
        ##

        # Applying the weights and summing along the last dimension for an output equal in size to the input tensors of
        # coordinates:
        return torch.einsum('i,...i->...', weights.to(device=self.device), samples)


class SinogramHEALPix(Sinogram):
    @staticmethod
    def build_grid(*, n_side: int, r_range: LinearRange, r_count: int, device=torch.device("cpu")) -> Sinogram3dGrid:
        raise Exception("Not yet implemented")

    def __init__(self, data: torch.Tensor, r_range: LinearRange):
        self.data = data
        self.r_range = r_range

    def to(self, **kwargs) -> 'SinogramHEALPix':
        return SinogramHEALPix(self.data.to(**kwargs), self.r_range)

    @property
    def device(self):
        return self.data.device

    def resample(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid) -> torch.Tensor:
        raise Exception("Not yet implemented")

    def resample_python(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid, *,
                        plot: bool = False) -> torch.Tensor:
        assert fixed_image_grid.device_consistent()
        assert fixed_image_grid.phi.device == self.device
        assert ph_matrix.device == self.device

        fixed_image_grid_sph = geometry.fixed_polar_to_moving_spherical(
            fixed_image_grid, ph_matrix=ph_matrix, plot=plot)
        raise Exception("Not yet implemented")


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
