import torch

from registration.lib.structs import *
import registration.lib.grangeat as grangeat


def calculate_volume_sinogram(cache_directory: str, volume_data: torch.Tensor, voxel_spacing: torch.Tensor,
                              ct_volume_path: str, volume_downsample_factor: int, *, device=torch.device('cpu'),
                              save_to_cache=True, vol_counts=256):
    print("Calculating 3D sinogram (the volume to resample)...")

    vol_diag: float = (voxel_spacing * torch.tensor(volume_data.size(), dtype=torch.float32,
                                                    device=voxel_spacing.device)).square().sum().sqrt().item()
    sinogram3d_range = Sinogram3dRange(LinearRange(-.5 * torch.pi, torch.pi * (.5 - 1. / float(vol_counts))),
                                       LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                       LinearRange(-.5 * vol_diag, .5 * vol_diag))

    sinogram3d_grid = sinogram3d_range.generate_linear_grid(vol_counts, device=device)
    sinogram3d = grangeat.calculate_radon_volume(volume_data, voxel_spacing=voxel_spacing, output_grid=sinogram3d_grid,
                                                 samples_per_direction=vol_counts)

    if save_to_cache:
        torch.save(VolumeSpec(ct_volume_path, volume_downsample_factor, sinogram3d, sinogram3d_range),
                   cache_directory + "/volume_spec.pt")

    print("Done and saved.")

    # X, Y, Z = torch.meshgrid(  #     [torch.arange(0, vol_counts, 1), torch.arange(0, vol_counts, 1), torch.arange(0, vol_counts, 1)])  # fig = pgo.Figure(data=pgo.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=sinogram3d.cpu().flatten(),  #                                  isomin=sinogram3d.min().item(), isomax=sinogram3d.max().item(), opacity=.2, surface_count=21))  # fig.show()

    # vol_image = ScalarImage(tensor=vol_data[None, :, :, :])  # vol_subject = read(vol_image, spacing=voxel_spacing)  # I believe that the detector array lies on the x-z plane, with x down, and z to the left (and so y outward)  # drr_generator = DRR(vol_subject,  # An object storing the CT volume, origin, and voxel spacing  #                     sdd=source_distance,  # Source-to-detector distance (i.e., focal length)  #                     height=int(torch.ceil(  #                         1.1 * voxel_spacing.mean() * torch.tensor(vol_size).max() / detector_spacing).item()),  #                     # Image height (if width is not provided, the generated DRR is square)  #                     delx=detector_spacing,  # Pixel spacing (in mm)  #                     ).to(device)  #

    return sinogram3d, sinogram3d_range
