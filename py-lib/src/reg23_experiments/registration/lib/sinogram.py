import gc
from abc import ABC, abstractmethod
from typing import NamedTuple, TypeVar, Tuple
import logging

import torch
import matplotlib.pyplot as plt

import reg23
from reg23_experiments.registration.lib.structs import SceneGeometry, Transformation, Sinogram2dGrid, LinearMapping, \
    LinearRange, Sinogram3dGrid
from reg23_experiments.registration.lib import geometry
from reg23_experiments.registration.lib import plot as myplt

__all__ = ["SinogramType", "Sinogram", "SinogramClassic", "SinogramHEALPix", "VolumeSpec", "DrrSpec"]

logger = logging.getLogger(__name__)

SinogramType = TypeVar('SinogramType')


class Sinogram(ABC):
    @abstractmethod
    def to(self, **kwargs) -> 'Sinogram':
        pass

    @property
    @abstractmethod
    def device(self):
        pass

    @property
    @abstractmethod
    def data(self):
        pass

    @property
    @abstractmethod
    def r_range(self):
        pass

    @abstractmethod
    def resample(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid) -> torch.Tensor:
        pass

    @abstractmethod
    def resample_cuda_texture(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid) -> torch.Tensor:
        pass

    @abstractmethod
    def resample_python(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid) -> torch.Tensor:
        pass


class SinogramClassic(Sinogram):
    phi_range = LinearRange(-0.5 * torch.pi, 0.5 * torch.pi)
    theta_range = LinearRange(-0.5 * torch.pi, 0.5 * torch.pi)

    @staticmethod
    def build_grid(*, counts: int | Tuple[int, int, int] | torch.Size, r_range: LinearRange,
                   device=torch.device("cpu")) -> Sinogram3dGrid:
        if isinstance(counts, int):
            counts = (int(torch.ceil(float(counts) * torch.tensor(0.5 * torch.pi).sqrt())),
                      int(torch.ceil(float(counts) * torch.tensor(0.5 * torch.pi).sqrt())), counts)
        elif isinstance(counts, torch.Size):
            assert len(counts) == 3
        phis = SinogramClassic.phi_range.generate_tex_coord_grid(counts[0], device=device)
        thetas = SinogramClassic.theta_range.generate_tex_coord_grid(counts[1], device=device)
        rs = r_range.generate_tex_coord_grid(counts[2], device=device)
        phis, thetas, rs = torch.meshgrid(phis, thetas, rs)
        return Sinogram3dGrid(phis, thetas, rs)

    def __init__(self, data: torch.Tensor, r_range: LinearRange, *, pad: bool = True):
        assert len(data.size()) == 3
        # dimensions should be (phi, theta, r) in decreasing size
        self._data = data
        self._r_range = r_range

        if pad:
            # cant wrap in theta or phi directions, so we pad:
            at_phi_left = self._data[0, :, :].unsqueeze(0)
            at_phi_right = self._data[-1, :, :].unsqueeze(0)
            left_padding = at_phi_right.flip(dims=(1, 2))  # flipped in r and theta directions
            right_padding = at_phi_left.flip(dims=(1, 2))  # flipped in r and theta directions
            self._data = torch.cat((left_padding, self._data, right_padding), dim=0)
            at_theta_top = self._data[:, 0, :].unsqueeze(1)
            at_theta_bot = self._data[:, -1, :].unsqueeze(1)
            top_padding = at_theta_bot.flip(dims=(2,))  # flipped in r direction
            bot_padding = at_theta_top.flip(dims=(2,))  # flipped in r direction
            self._data = torch.cat((top_padding, self._data, bot_padding), dim=1)

        logger.info(
            "SinogramClassic initialised with size (phi count + 2, theta count + 2, r count) = ({}, {}, {})".format(
                self._data.size()[0], self._data.size()[1], self._data.size()[2]))

        if self.device.type == "cuda":
            self._texture = reg23.CUDATexture3D(self.data, "zero", "zero", "zero")

    def __getstate__(self):
        return {"_data": self._data, "_r_range": self._r_range}

    def __setstate__(self, state):
        self._data = state["_data"]
        self._r_range = state["_r_range"]
        if self.device.type == "cuda":
            self._texture = reg23.CUDATexture3D(self.data, "zero", "zero", "zero")

    @property
    def device(self):
        return self.data.device

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def r_range(self) -> LinearRange:
        return self._r_range

    @property
    def grid(self) -> Sinogram3dGrid:
        return SinogramClassic.build_grid(
            counts=(self.data.size()[0] - 2, self.data.size()[1] - 2, self.data.size()[2]), r_range=self.r_range,
            device=self.device)  # the '-2's adjust for padding

    def to(self, **kwargs) -> 'SinogramClassic':
        return SinogramClassic(self.data.to(**kwargs), self.r_range, pad=False)

    def resample(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid, *,
                 out: torch.Tensor | None = None) -> torch.Tensor:
        return reg23.resample_sinogram3d(self.data, "classic", self.r_range.get_spacing(self.data.size()[2]), ph_matrix,
                                         fixed_image_grid.phi, fixed_image_grid.r, out=out)

    def resample_cuda_texture(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid, *,
                              out: torch.Tensor | None = None) -> torch.Tensor:
        assert self.device.type == "cuda"
        return reg23.resample_sinogram3d_cuda_texture(self._texture, "classic",
                                                      self.r_range.get_spacing(self.data.size()[2]), ph_matrix,
                                                      fixed_image_grid.phi, fixed_image_grid.r, out=out)

    def resample_python(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid, *, smooth: float | None = None,
                        plot: bool = False, out: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param ph_matrix:
        :param fixed_image_grid:
        :param smooth:
        :param plot:
        :param out:
        :return:
        """
        assert fixed_image_grid.device_consistent()
        assert fixed_image_grid.phi.device == self.device
        assert ph_matrix.device == self.device

        fixed_image_grid_sph = geometry.fixed_polar_to_moving_spherical(fixed_image_grid, ph_matrix=ph_matrix,
                                                                        plot=plot)

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
        theta_grid_padding_offsets: float = 2.0 / float(self._data.size()[1])
        theta_grid_sample_range_padded = LinearRange(-1. + theta_grid_padding_offsets, 1. - theta_grid_padding_offsets)
        j_mapping: LinearMapping = theta_grid_sample_range_padded.get_mapping_from(self.theta_range)
        phi_grid_padding_offsets: float = 2.0 / float(self._data.size()[0])
        phi_grid_sample_range_padded = LinearRange(-1. + phi_grid_padding_offsets, 1. - phi_grid_padding_offsets)
        k_mapping: LinearMapping = phi_grid_sample_range_padded.get_mapping_from(self.phi_range)

        if out is None:
            out = torch.zeros_like(fixed_image_grid_sph.phi)
        grid = torch.stack((i_mapping(fixed_image_grid_sph.r), j_mapping(fixed_image_grid_sph.theta),
                            k_mapping(fixed_image_grid_sph.phi)), dim=-1)
        ret = reg23.grid_sample3d(self.data, grid, "zero", "zero", "zero", out=out)

        del fixed_image_grid_sph

        # sign changes - this implementation relies on the convenient coordinate system
        moving_origin_projected = ph_matrix[0:2, 3] / ph_matrix[3, 3]
        square_radius: torch.Tensor = .25 * moving_origin_projected.square().sum()
        need_sign_change = ((fixed_image_grid.r.unsqueeze(-1) * torch.stack(
            (torch.cos(fixed_image_grid.phi), torch.sin(fixed_image_grid.phi)),
            dim=-1) - .5 * moving_origin_projected).square().sum(dim=-1) < square_radius)
        #

        ret[need_sign_change] *= -1.

        del need_sign_change

        return ret


"""
    def grid_sample_smoothed(self, grid: Sinogram3dGrid, *, i_mapping: LinearMapping, j_mapping: LinearMapping,
                             k_mapping: LinearMapping, sigma: float, offset_count: int = 10):
        
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
        samples = reg23.grid_sample3d(self.data, grid, address_mode="wrap")

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
    """


class SinogramHEALPix(Sinogram):
    @staticmethod
    def spherical_to_tex_coord(spherical_grid: Sinogram3dGrid, n_side: float) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        # to x_s, y_s
        z = spherical_grid.theta.sin()  # sin instead of cos for adjustment
        z_abs = z.abs()
        sigma = z.sign() * (2.0 - (3.0 * (1.0 - z_abs)).sqrt())
        equatorial_zone = z_abs <= 2. / 3.
        polar_caps = torch.logical_not(equatorial_zone)
        x_s = torch.zeros_like(z)
        x_s[equatorial_zone] = spherical_grid.phi[equatorial_zone] + 0.5 * torch.pi  # with pi/2 adjustment
        x_s[polar_caps] = (spherical_grid.phi + 0.5 * torch.pi - (sigma.abs() - 1.0) * (
                torch.fmod(spherical_grid.phi + 0.5 * torch.pi, 0.5 * torch.pi) - 0.25 * torch.pi))[
            polar_caps]  # with pi/2 adjustment
        y_s = torch.zeros_like(z)
        y_s[equatorial_zone] = 3.0 * torch.pi * z[equatorial_zone] / 8.0
        y_s[polar_caps] = torch.pi * sigma[polar_caps] / 4.0
        del equatorial_zone, polar_caps, sigma, z, z_abs

        # _, axes = plt.subplots()
        # mesh = axes.pcolormesh(spherical_grid.phi.numpy(), spherical_grid.theta.numpy(), x_s.numpy())
        # plt.colorbar(mesh)
        # axes.set_xlabel("phi")
        # axes.set_ylabel("theta")
        # axes.set_title("x_s")
        #
        # _, axes = plt.subplots()
        # mesh = axes.pcolormesh(spherical_grid.phi.numpy(), spherical_grid.theta.numpy(), y_s.numpy())
        # plt.colorbar(mesh)
        # axes.set_xlabel("phi")
        # axes.set_ylabel("theta")
        # axes.set_title("y_s")

        # to x_p, y_p
        x_p = 2.0 * n_side * x_s / torch.pi
        y_p = n_side * (1.0 - 2.0 * y_s / torch.pi)
        del x_s, y_s

        # _, axes = plt.subplots()
        # mesh = axes.pcolormesh(spherical_grid.phi.numpy(), spherical_grid.theta.numpy(), x_p.numpy())
        # plt.colorbar(mesh)
        # axes.set_xlabel("phi")
        # axes.set_ylabel("theta")
        # axes.set_title("x_p")
        #
        # _, axes = plt.subplots()
        # mesh = axes.pcolormesh(spherical_grid.phi.numpy(), spherical_grid.theta.numpy(), y_p.numpy())
        # plt.colorbar(mesh)
        # axes.set_xlabel("phi")
        # axes.set_ylabel("theta")
        # axes.set_title("y_p")

        # to u, v
        u = x_p - y_p + 1.5 * n_side
        v = x_p + y_p - 0.5 * n_side
        v_high = v >= 2.0 * n_side
        u_high = u >= 2.0 * n_side
        base_pixel_6 = torch.logical_and(v_high, u_high)
        u[base_pixel_6] -= 2.0 * n_side
        v[v_high] -= 2.0 * n_side

        # flipping base pixel 6 upside-down
        temp = u[base_pixel_6]
        u[base_pixel_6] = v[base_pixel_6]
        v[base_pixel_6] = temp
        del temp
        r = spherical_grid.r.clone()
        r[base_pixel_6] *= -1.0  # r is flipped for base pixel 6

        base_pixel_9 = torch.logical_and(v_high, torch.logical_not(u_high))
        u[base_pixel_9] += n_side + 2.0  # the 2 adjusts for padding
        v[base_pixel_9] -= 2.0  # this adjusts for padding

        return u + 1.0, v + 3.0, r  # the 1 and 3 adjust for padding

    @staticmethod
    def tex_coord_to_spherical(u: torch.Tensor, v: torch.Tensor, r: torch.Tensor, n_side: float) -> Sinogram3dGrid:
        assert u.size() == v.size()

        u_star = u.clone()
        v_star = v.clone()
        base_pixel_9 = torch.logical_and(u_star >= 2.0 * n_side + 2.0,
                                         v_star < n_side + 2.0)  # the added 2s adjust for padding
        u_star -= 1.0  # this adjusts for padding
        v_star -= 3.0  # this adjusts for padding
        u_star[base_pixel_9] -= 2.0 + n_side  # the 2 adjusts for padding
        v_star[base_pixel_9] += 2.0  # this adjusts for padding
        base_pixel_6 = u_star + v_star < n_side
        u_star[base_pixel_6] += 2.0 * n_side
        v_star[torch.logical_or(base_pixel_9, base_pixel_6)] += 2.0 * n_side
        del base_pixel_9
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        x_p = 0.5 * (u_star + v_star - n_side)
        y_p = 0.5 * (v_star - u_star) + n_side
        del u_star, v_star
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        x_s = 0.5 * torch.pi * x_p / n_side
        y_s = 0.5 * torch.pi * (1.0 - y_p / n_side)
        del x_p, y_p
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        y_s[base_pixel_6] *= -1.0  # theta is flipped for base pixel 6
        r_ = r.clone()
        r_[base_pixel_6] *= -1.0  # r is flipped for base pixel 6
        del base_pixel_6
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # to phi, theta
        y_s_abs = y_s.abs()
        equatorial_zone = y_s_abs <= 0.25 * torch.pi
        polar_caps = torch.logical_not(equatorial_zone)
        z = torch.zeros_like(x_s)
        z[equatorial_zone] = (8.0 * y_s / (3.0 * torch.pi))[equatorial_zone]
        z[polar_caps] = ((1.0 - (2.0 - 4.0 * y_s_abs / torch.pi).square() / 3.0) * y_s.sign())[polar_caps]

        phi = torch.zeros_like(x_s)
        phi[equatorial_zone] = x_s[equatorial_zone] - 0.5 * torch.pi  # with pi/2 adjustment
        phi[polar_caps] = (x_s - (torch.fmod(x_s, 0.5 * torch.pi) - 0.25 * torch.pi) * (y_s_abs - 0.25 * torch.pi) / (
                y_s_abs - 0.5 * torch.pi - 1e-5))[polar_caps] - 0.5 * torch.pi  # with pi/2 adjustment
        del x_s, y_s, y_s_abs, equatorial_zone, polar_caps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        theta = z.asin()  # instead of cos, for adjustment
        del z
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return Sinogram3dGrid(phi=phi, theta=theta, r=r_)

    @staticmethod
    def build_grid(*, n_side: int, r_range: LinearRange, r_count: int, device=torch.device("cpu")) -> Sinogram3dGrid:
        u = LinearRange(0.0, 3.0 * float(n_side)).generate_tex_coord_grid(3 * n_side, device=device)
        v = LinearRange(0.0, 2.0 * float(n_side)).generate_tex_coord_grid(2 * n_side, device=device)
        r = r_range.generate_grid(r_count, device=device)
        r, v, u = torch.meshgrid(r, v, u)
        u = u.clone()
        v = v.clone()
        r = r.clone()
        base_pixel_9 = torch.logical_and(u >= 2.0 * float(n_side), v < float(n_side))
        u += 1.0
        v += 3.0
        u[base_pixel_9] += 2.0
        v[base_pixel_9] -= 2.0

        return SinogramHEALPix.tex_coord_to_spherical(u, v, r, n_side)

    def __init__(self, data: torch.Tensor, r_range: LinearRange, pad: bool = True):
        assert len(data.size()) == 3
        if pad:
            # size is r, v, u
            assert data.size()[2] % 3 == 0
            assert data.size()[1] % 2 == 0
            assert data.size()[2] // 3 == data.size()[1] // 2
            n_side: int = data.size()[1] // 2

            self._data = data

            bp_9 = self._data[:, :n_side, (2 * n_side):]

            bp_0_top_left = self._data[:, 0, n_side:(2 * n_side)].unsqueeze(1)
            bp_0_top_right = self._data[:, :n_side, 2 * n_side - 1].unsqueeze(1)
            bp_1_top_left = self._data[:, n_side, (2 * n_side):].unsqueeze(1)
            bp_1_top_right = self._data[:, n_side:, -1].unsqueeze(1)
            bp_1_bot_right = self._data[:, -1, (2 * n_side):].unsqueeze(1)
            bp_5_bot_right = self._data[:, -1, n_side:(2 * n_side)].unsqueeze(1)
            bp_8_bot_right = self._data[:, -1, :n_side].unsqueeze(1)
            bp_8_bot_left = self._data[:, n_side:, 0].unsqueeze(1)
            bp_9_top_left = self._data[:, 0, (2 * n_side):].unsqueeze(1)
            bp_9_top_right = self._data[:, :n_side, -1].unsqueeze(1)
            bp_9_bot_right = self._data[:, n_side - 1, (2 * n_side):].unsqueeze(1)
            bp_9_bot_left = self._data[:, :n_side, 2 * n_side].unsqueeze(1)
            bp_6_top_left = self._data[:, 0, :n_side].unsqueeze(1)
            bp_6_bot_left = self._data[:, :n_side, 0].unsqueeze(1)

            pad_0_top_left = bp_9_bot_right.flip(dims=(0, -1))  # wraps causing flip in theta and r
            pad_0_top_right = bp_1_top_left.flip(dims=(-1,))
            pad_1_top_left = bp_0_top_right.flip(dims=(-1,))
            pad_1_top_right = bp_8_bot_left.flip(dims=(0, -1))  # wraps causing flip in theta and r
            pad_1_bot_right = bp_6_bot_left.flip(
                dims=(0,))  # other side of 6 because 6 is flipped, flipped in r because so is 6
            pad_5_bot_right = bp_9_top_left
            pad_8_bot_right = bp_9_bot_left.flip(dims=(-1,))
            pad_8_bot_left = bp_1_top_right.flip(dims=(0, -1))  # wraps causing flip in theta and r
            pad_6_bot_left = bp_1_bot_right.flip(
                dims=(0,))  # other side of 6 because 6 is flipped, flipped in r because so is 6
            pad_6_top_left = bp_9_top_right.flip(
                dims=(0,))  # other side of 6 because 6 is flipped, flipped in r because so is 6

            pad_9_top_left = bp_5_bot_right
            pad_9_top_right = bp_6_top_left.flip(
                dims=(0,))  # other side of 6 because 6 is flipped, flipped in r because so is 6
            pad_9_bot_right = bp_0_top_left.flip(dims=(0, -1))  # wraps causing flip in theta and r
            pad_9_bot_left = bp_8_bot_right.flip(dims=(-1,))

            del bp_0_top_left, bp_0_top_right, bp_1_top_left, bp_1_top_right, bp_1_bot_right, bp_5_bot_right, (
                bp_8_bot_right), bp_8_bot_left, bp_9_top_left, bp_9_top_right, bp_9_bot_right, bp_9_bot_left, (
                bp_6_top_left), bp_6_bot_left

            corner_5_right = self._data[:, 2 * n_side - 1, 2 * n_side - 1].unsqueeze(1).unsqueeze(1)
            corner_0_top = self._data[:, 0, 2 * n_side - 1].unsqueeze(1).unsqueeze(1)
            corner_0_left = self._data[:, 0, n_side].unsqueeze(1).unsqueeze(1)
            corner_1_top = self._data[:, n_side, -1].unsqueeze(1).unsqueeze(1)
            corner_1_bot = self._data[:, -1, 2 * n_side].unsqueeze(1).unsqueeze(1)
            corner_8_bot = self._data[:, -1, 0].unsqueeze(1).unsqueeze(1)
            corner_8_left = self._data[:, n_side, 0].unsqueeze(1).unsqueeze(1)
            corner_8_right = self._data[:, -1, n_side - 1].unsqueeze(1).unsqueeze(1)
            corner_9_bot = self._data[:, n_side - 1, 2 * n_side].unsqueeze(1).unsqueeze(1)

            pad_6_corner_left = corner_5_right.flip(dims=(0,))  # flipped in r because so is 6
            pad_0_corner_top = corner_8_bot.flip(dims=(0,))  # wraps causing flip in r
            pad_1_corner_top = corner_9_bot.flip(dims=(0,))  # wraps causing flip in r
            pad_1_corner_right = corner_8_left.flip(dims=(0,))  # wraps causing flip in r
            pad_8_corner_bot = corner_0_top.flip(dims=(0,))  # wraps causing flip in r
            pad_9_corner_left = corner_8_right
            pad_9_corner_top = corner_1_bot
            pad_9_corner_right = corner_0_left.flip(dims=(0,))  # wraps causing flip in r
            pad_9_corner_bot = corner_1_top.flip(dims=(0,))  # wraps causing flip in r

            del corner_5_right, corner_0_top, corner_0_left, corner_1_top, corner_1_bot, corner_8_bot, corner_8_left, corner_8_right, corner_9_bot

            r_count = data.size()[0]
            row_0 = torch.cat((torch.zeros(r_count, 1, 2 * n_side + 2, device=self.device),  #
                               pad_9_corner_left,  #
                               pad_9_top_left,  #
                               pad_9_corner_top), dim=-1)
            row_1 = torch.cat((torch.zeros(r_count, 1, 2 * n_side + 2, device=self.device),  #
                               pad_9_bot_left[:, :, 0].unsqueeze(1),  #
                               bp_9[:, 0, :].unsqueeze(1),  #
                               pad_9_top_right[:, :, 0].unsqueeze(1)), dim=-1)
            row_2 = torch.cat((pad_6_corner_left,  #
                               pad_6_top_left,  #
                               pad_0_top_left,  #
                               pad_0_corner_top,  #
                               pad_9_bot_left[:, :, 1].unsqueeze(1),  #
                               bp_9[:, 1, :].unsqueeze(1),  #
                               pad_9_top_right[:, :, 1].unsqueeze(1)), dim=-1)
            rows_3_to_n = torch.cat((pad_6_bot_left[:, :, :-2].transpose(1, 2),  #
                                     self._data[:, :(n_side - 2), :(2 * n_side)],  #
                                     pad_0_top_right[:, :, :-2].transpose(1, 2),  #
                                     pad_9_bot_left[:, :, 2:].transpose(1, 2),  #
                                     bp_9[:, 2:, :],  #
                                     pad_9_top_right[:, :, 2:].transpose(1, 2)), dim=-1)
            row_np1 = torch.cat((pad_6_bot_left[:, :, -2].unsqueeze(1),  #
                                 self._data[:, n_side - 2, :(2 * n_side)].unsqueeze(1),  #
                                 pad_0_top_right[:, :, -2].unsqueeze(1),  #
                                 pad_9_corner_bot,  #
                                 pad_9_bot_right,  #
                                 pad_9_corner_right), dim=-1)
            row_np2 = torch.cat((pad_6_bot_left[:, :, -1].unsqueeze(1),  #
                                 self._data[:, n_side - 1, :(2 * n_side)].unsqueeze(1),  #
                                 pad_1_top_left,  #
                                 pad_1_corner_top,  #
                                 torch.zeros(r_count, 1, 2, device=self.device)), dim=-1)
            rows_np3_to_2np2 = torch.cat((pad_8_bot_left.transpose(1, 2),  #
                                          self._data[:, n_side:, :],  #
                                          pad_1_top_right.transpose(1, 2),  #
                                          torch.zeros(r_count, n_side, 2, device=self.device)), dim=-1)
            row_2np3 = torch.cat((pad_8_corner_bot,  #
                                  pad_8_bot_right,  #
                                  pad_5_bot_right,  #
                                  pad_1_bot_right,  #
                                  pad_1_corner_right,  #
                                  torch.zeros(r_count, 1, 2, device=self.device)), dim=-1)

            del pad_6_corner_left, pad_0_corner_top, pad_1_corner_top, pad_1_corner_right, pad_8_corner_bot, (
                pad_9_corner_left), pad_9_corner_top, pad_9_corner_right, pad_9_corner_bot

            self._data = torch.cat((row_0, row_1, row_2, rows_3_to_n, row_np1, row_np2, rows_np3_to_2np2, row_2np3),
                                   dim=1)

            del row_0, row_1, row_2, rows_3_to_n, row_np1, row_np2, rows_np3_to_2np2, row_2np3
        else:
            assert (data.size()[2] - 4) % 3 == 0
            assert (data.size()[1] - 4) % 2 == 0
            assert (data.size()[2] - 4) // 3 == (data.size()[1] - 4) // 2
            self._data = data

        logger.info("SinogramHEALPix initialised with size (r count, 2*N_side + 4, 3*N_side + 4) = ({}, {}, {})".format(
            self._data.size()[0], self._data.size()[1], self._data.size()[2]))

        self._r_range = r_range

        if self.device.type == "cuda":
            self._texture = reg23.CUDATexture3D(self.data, "zero", "zero", "zero")

    def __getstate__(self):
        return {"_data": self._data, "_r_range": self._r_range}

    def __setstate__(self, state):
        self._data = state["_data"]
        self._r_range = state["_r_range"]
        if self.device.type == "cuda":
            self._texture = reg23.CUDATexture3D(self.data, "zero", "zero", "zero")

    def to(self, **kwargs) -> 'SinogramHEALPix':
        return SinogramHEALPix(self.data.to(**kwargs), self.r_range, pad=False)

    @property
    def device(self):
        return self.data.device

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def r_range(self) -> LinearRange:
        return self._r_range

    def sample(self, grid: Sinogram3dGrid, out: torch.Tensor | None = None) -> torch.Tensor:
        assert grid.device_consistent()
        assert grid.phi.device == self.device
        if out is not None:
            assert out.device == self.device

        n_side: int = (self.data.size()[1] - 4) // 2

        u, v, r = SinogramHEALPix.spherical_to_tex_coord(grid, float(n_side))

        # texCoord is in the reverse order: (X, Y, Z)
        grid_range = LinearRange.grid_sample_range()
        i_mapping: LinearMapping = grid_range.get_mapping_from(LinearRange(low=0., high=float(self._data.size()[2])))
        j_mapping: LinearMapping = grid_range.get_mapping_from(LinearRange(low=0., high=float(self._data.size()[1])))
        k_mapping: LinearMapping = grid_range.get_mapping_from(self.r_range)

        if out is None:
            out = torch.zeros_like(grid.phi)
        grid = torch.stack((i_mapping(u), j_mapping(v), k_mapping(r)), dim=-1)
        return reg23.grid_sample3d(self.data, grid, "zero", "zero", "zero", out=out)

    def resample(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid, *,
                 out: torch.Tensor | None = None) -> torch.Tensor:
        return reg23.resample_sinogram3d(self.data, "healpix", self.r_range.get_spacing(self.data.size()[0]), ph_matrix,
                                         fixed_image_grid.phi, fixed_image_grid.r, out=out)

    def resample_cuda_texture(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid, *,
                              out: torch.Tensor | None = None) -> torch.Tensor:
        assert self.device.type == "cuda"
        return reg23.resample_sinogram3d_cuda_texture(self._texture, "healpix",
                                                      self.r_range.get_spacing(self.data.size()[0]), ph_matrix,
                                                      fixed_image_grid.phi, fixed_image_grid.r, out=out)

    def resample_python(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid, *, plot: bool = False,
                        out: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param ph_matrix:
        :param fixed_image_grid:
        :param plot:
        :param out:
        :return:
        """
        assert fixed_image_grid.device_consistent()
        assert fixed_image_grid.phi.device == self.device
        assert ph_matrix.device == self.device

        fixed_image_grid_sph = geometry.fixed_polar_to_moving_spherical(fixed_image_grid, ph_matrix=ph_matrix,
                                                                        plot=plot)

        if out is None:
            out = torch.zeros_like(fixed_image_grid_sph.phi)
        ret = self.sample(fixed_image_grid_sph, out=out)
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


# class SinogramFibonacci(Sinogram):
#     def __init__(self, data: torch.Tensor, r_range: LinearRange):
#         self.data = data
#         self.r_range = r_range
#
#     def to(self, **kwargs) -> 'SinogramFibonacci':
#         return SinogramFibonacci(self.data.to(**kwargs), self.r_range)
#
#     def device(self):
#         return self.data.device
#
#     def resample(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid) -> torch.Tensor:
#         raise Exception("Not yet implemented")
#
#     def resample_python(self, ph_matrix: torch.Tensor, fixed_image_grid: Sinogram2dGrid) -> torch.Tensor:
#         raise Exception("Not yet implemented")


class VolumeSpec(NamedTuple):
    ct_volume_path: str
    downsample_factor: int
    sinogram: Sinogram


class DrrSpec(NamedTuple):
    ct_volume_path: str
    detector_spacing: torch.Tensor  # [mm] distances between the detectors: (vertical, horizontal)
    scene_geometry: SceneGeometry
    image: torch.Tensor
    transformation: Transformation
