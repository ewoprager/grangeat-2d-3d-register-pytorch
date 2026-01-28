import torch

from backend import reg23

__all__ = ["radon2d", "radon2d_v2", "d_radon2d_dr", "radon3d", "radon3d_v2", "d_radon3d_dr", "d_radon3d_dr_v2",
           "resample_sinogram3d", "normalised_cross_correlation", "grid_sample3d", "project_drr",
           "project_drr_cuboid_mask"]

if torch.cuda.is_available():
    from . import structs

    __all__ += ["resample_sinogram3d_cuda_texture", "project_drrs_batched"]


def radon2d(image: torch.Tensor, image_spacing: torch.Tensor, phi_values: torch.Tensor, r_values: torch.Tensor,
            samples_per_line: int) -> torch.Tensor:
    return torch.ops.reg23.radon2d.default(image, image_spacing.to(dtype=torch.float64), phi_values, r_values,
                                           samples_per_line)


def radon2d_v2(image: torch.Tensor, image_spacing: torch.Tensor, phi_values: torch.Tensor, r_values: torch.Tensor,
               samples_per_line: int) -> torch.Tensor:
    """
    Note: CPU implementation is identical to that of `radon2d`
    """
    return torch.ops.reg23.radon2d_v2.default(image, image_spacing.to(dtype=torch.float64), phi_values, r_values,
                                              samples_per_line)


def d_radon2d_dr(image: torch.Tensor, image_spacing: torch.Tensor, phi_values: torch.Tensor, r_values: torch.Tensor,
                 samples_per_line: int) -> torch.Tensor:
    return torch.ops.reg23.d_radon2d_dr.default(image, image_spacing.to(dtype=torch.float64), phi_values, r_values,
                                                samples_per_line)


def radon3d(volume: torch.Tensor, volume_spacing: torch.Tensor, phi_values: torch.Tensor, theta_values: torch.Tensor,
            r_values: torch.Tensor, samples_per_direction: int) -> torch.Tensor:
    return torch.ops.reg23.radon3d.default(volume, volume_spacing.to(dtype=torch.float64), phi_values, theta_values,
                                           r_values, samples_per_direction)


def radon3d_v2(volume: torch.Tensor, volume_spacing: torch.Tensor, phi_values: torch.Tensor, theta_values: torch.Tensor,
               r_values: torch.Tensor, samples_per_direction: int) -> torch.Tensor:
    """
    Note: CPU implementation is identical to that of `radon3d`
    """
    return torch.ops.reg23.radon3d_v2.default(volume, volume_spacing.to(dtype=torch.float64), phi_values, theta_values,
                                              r_values, samples_per_direction)


def d_radon3d_dr(volume: torch.Tensor, volume_spacing: torch.Tensor, phi_values: torch.Tensor,
                 theta_values: torch.Tensor, r_values: torch.Tensor, samples_per_direction: int) -> torch.Tensor:
    return torch.ops.reg23.d_radon3d_dr.default(volume, volume_spacing.to(dtype=torch.float64), phi_values,
                                                theta_values, r_values, samples_per_direction)


def d_radon3d_dr_v2(volume: torch.Tensor, volume_spacing: torch.Tensor, phi_values: torch.Tensor,
                    theta_values: torch.Tensor, r_values: torch.Tensor, samples_per_direction: int) -> torch.Tensor:
    return torch.ops.reg23.d_radon3d_dr_v2.default(volume, volume_spacing.to(dtype=torch.float64), phi_values,
                                                   theta_values, r_values, samples_per_direction)


def resample_sinogram3d(sinogram3d: torch.Tensor, sinogram_type: str, r_spacing: float, projection_matrix: torch.Tensor,
                        phi_values: torch.Tensor, r_values: torch.Tensor,
                        out: torch.Tensor | None = None) -> torch.Tensor:
    return torch.ops.reg23.resample_sinogram3d.default(sinogram3d, sinogram_type, r_spacing, projection_matrix,
                                                       phi_values, r_values, out)


if torch.cuda.is_available():
    def resample_sinogram3d_cuda_texture(texture: structs.CUDATexture3D, sinogram_type: str, r_spacing: float,
                                         projection_matrix: torch.Tensor, phi_values: torch.Tensor,
                                         r_values: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        size = texture.size
        return torch.ops.reg23.resample_sinogram3d_cuda_texture.default(texture.handle, size[0], size[1], size[2],
                                                                        sinogram_type, r_spacing, projection_matrix,
                                                                        phi_values, r_values, out)


def normalised_cross_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ret, *_ = torch.ops.reg23.normalised_cross_correlation.default(a, b)
    return ret


def grid_sample3d(input_: torch.Tensor, grid: torch.Tensor, address_mode_x: str = "zero", address_mode_y: str = "zero",
                  address_mode_z: str = "zero", out: torch.Tensor | None = None) -> torch.Tensor:
    return torch.ops.reg23.grid_sample3d.default(input_, grid.to(dtype=torch.float32), address_mode_x, address_mode_y,
                                                 address_mode_z, out=out)


def project_drr(volume: torch.Tensor, voxel_spacing: torch.Tensor, homography_matrix_inverse: torch.Tensor,
                source_distance: float, output_width: int, output_height: int, output_offset: torch.Tensor,
                detector_spacing: torch.Tensor) -> torch.Tensor:
    return torch.ops.reg23.project_drr.default(volume, voxel_spacing.to(dtype=torch.float64),
                                               homography_matrix_inverse.to(dtype=torch.float64), source_distance,
                                               output_width, output_height, output_offset.to(dtype=torch.float64),
                                               detector_spacing.to(dtype=torch.float64))


def project_drr_backward(volume: torch.Tensor, voxel_spacing: torch.Tensor, homography_matrix_inverse: torch.Tensor,
                         source_distance: float, output_width: int, output_height: int, output_offset: torch.Tensor,
                         detector_spacing: torch.Tensor, d_loss_d_drr: torch.Tensor) -> torch.Tensor:
    return torch.ops.reg23.project_drr_backward.default(volume, voxel_spacing.to(dtype=torch.float64),
                                                        homography_matrix_inverse.to(dtype=torch.float64),
                                                        source_distance, output_width, output_height, output_offset,
                                                        detector_spacing.to(dtype=torch.float64),
                                                        d_loss_d_drr.to(dtype=torch.float32))


def project_drr_cuboid_mask(volume_size: torch.Tensor, voxel_spacing: torch.Tensor,
                            homography_matrix_inverse: torch.Tensor, source_distance: float, output_width: int,
                            output_height: int, output_offset: torch.Tensor,
                            detector_spacing: torch.Tensor) -> torch.Tensor:
    return torch.ops.reg23.project_drr_cuboid_mask.default(volume_size, voxel_spacing.to(dtype=torch.float64),
                                                           homography_matrix_inverse.to(dtype=torch.float64),
                                                           source_distance, output_width, output_height, output_offset,
                                                           detector_spacing.to(dtype=torch.float64))


if torch.cuda.is_available():
    def project_drrs_batched(volume: torch.Tensor, voxel_spacing: torch.Tensor, inverse_h_matrices: torch.Tensor,
                             source_distance: float, output_width: int, output_height: int, output_offset: torch.Tensor,
                             detector_spacing: torch.Tensor) -> torch.Tensor:
        return torch.ops.reg23.project_drrs_batched.default(volume, voxel_spacing.to(dtype=torch.float64),
                                                            inverse_h_matrices.to(dtype=torch.float64), source_distance,
                                                            output_width, output_height,
                                                            output_offset.to(dtype=torch.float64),
                                                            detector_spacing.to(dtype=torch.float64))
