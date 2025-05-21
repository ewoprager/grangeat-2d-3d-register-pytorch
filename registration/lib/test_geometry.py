import pytest
import torch

import matplotlib.pyplot as plt

import Extension as ExtensionTest
from registration.lib.geometry import *


def test_fixed_polar_to_moving_cartesian():
    device = torch.device('cpu')

    a = torch.tensor([10.], device=device)
    b = torch.tensor([1000.], device=device)
    hypotenuse = (a.square() + b.square()).sqrt()
    sin_alpha = a / hypotenuse
    cos_alpha = b / hypotenuse

    input_grid = Sinogram2dGrid(phi=torch.tensor([0.], device=device), r=a)
    scene_geometry = SceneGeometry(source_distance=b.item())
    transformation = Transformation(rotation=torch.zeros(3, device=device), translation=torch.zeros(3, device=device))
    source_position = scene_geometry.source_position(device=device)
    p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
    ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=device)).to(dtype=torch.float32)
    ret = fixed_polar_to_moving_cartesian(input_grid, ph_matrix=ph_matrix)
    assert isinstance(ret, torch.Tensor)
    assert ret.device == input_grid.phi.device
    assert ret.size() == torch.Size([1, 3])
    assert ret[0, 0].item() == pytest.approx((b * sin_alpha * cos_alpha).item(), abs=1e-4)
    assert ret[0, 1].item() == pytest.approx(0., abs=1e-4)
    assert ret[0, 2].item() == pytest.approx(b * sin_alpha.square().item(), abs=1e-4)

    transformation = Transformation(rotation=torch.zeros(3, device=device),
                                    translation=torch.tensor([b * sin_alpha * cos_alpha, 0., b * sin_alpha.square()]))
    source_position = scene_geometry.source_position(device=device)
    p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
    ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=device)).to(dtype=torch.float32)
    ret = fixed_polar_to_moving_cartesian(input_grid, ph_matrix=ph_matrix)
    assert ret[0, 0].item() == pytest.approx(0., abs=1e-4)
    assert ret[0, 1].item() == pytest.approx(0., abs=1e-4)
    assert ret[0, 2].item() == pytest.approx(0., abs=1e-4)

    angle = 0.32341
    input_grid = Sinogram2dGrid(phi=torch.tensor([angle], device=device), r=-a)
    transformation = Transformation(rotation=torch.zeros(3, device=device), translation=torch.zeros(3, device=device))
    source_position = scene_geometry.source_position(device=device)
    p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
    ph_matrix = torch.matmul(p_matrix, transformation.get_h(device=device)).to(dtype=torch.float32)
    ret_a = fixed_polar_to_moving_cartesian(input_grid, ph_matrix=ph_matrix)
    input_grid = Sinogram2dGrid(phi=torch.tensor([angle + torch.pi], device=device), r=a)
    ret_b = fixed_polar_to_moving_cartesian(input_grid, ph_matrix=ph_matrix)
    assert ret_a[0, 0].item() == pytest.approx(ret_b[0, 0].item(), abs=1e-4)
    assert ret_a[0, 1].item() == pytest.approx(ret_b[0, 1].item(), abs=1e-4)
    assert ret_a[0, 2].item() == pytest.approx(ret_b[0, 2].item(), abs=1e-4)

    input_grid = Sinogram2dGrid(phi=torch.tensor([.5 * torch.pi], device=device), r=a)
    ret = fixed_polar_to_moving_cartesian(input_grid, ph_matrix=ph_matrix)
    assert ret[0, 0].item() == pytest.approx(0., abs=1e-4)
    assert ret[0, 1].item() == pytest.approx((b * sin_alpha * cos_alpha).item(), abs=1e-4)
    assert ret[0, 2].item() == pytest.approx(b * sin_alpha.square().item(), abs=1e-4)


def test_moving_cartesian_to_moving_spherical():
    device = torch.device('cpu')
    ret = moving_cartesian_to_moving_spherical(torch.tensor([1., 0., 0.], device=device))
    assert isinstance(ret, Sinogram3dGrid)
    assert ret.device_consistent()
    assert ret.phi.device == device
    assert ret.size_consistent()
    assert ret.phi.size() == torch.Size([])
    assert ret.phi.item() == pytest.approx(0., abs=1e-4)
    assert ret.theta.item() == pytest.approx(0., abs=1e-4)
    assert ret.r.item() == pytest.approx(1., abs=1e-4)

    ret = moving_cartesian_to_moving_spherical(torch.tensor([[[1., 0., 0.]]], device=device))
    assert isinstance(ret, Sinogram3dGrid)
    assert ret.device_consistent()
    assert ret.phi.device == device
    assert ret.size_consistent()
    assert ret.phi.size() == torch.Size([1, 1])

    ret = moving_cartesian_to_moving_spherical(torch.tensor([0., 1., 0.], device=device))
    assert ret.phi.item() * ret.r.item() == pytest.approx(.5 * torch.pi, abs=1e-4)
    assert ret.theta.item() == pytest.approx(0., abs=1e-4)

    ret = moving_cartesian_to_moving_spherical(torch.tensor([0., 0., 1.], device=device))
    assert ret.theta.item() * ret.r.item() == pytest.approx(.5 * torch.pi, abs=1e-4)

    ret = moving_cartesian_to_moving_spherical(torch.tensor([-1., 0., 0.], device=device))
    assert ret.phi.item() == pytest.approx(0., abs=1e-4)
    assert ret.theta.item() == pytest.approx(0., abs=1e-4)
    assert ret.r.item() == pytest.approx(-1., abs=1e-4)


def test_generate_drr_python():
    device = torch.device('cpu')
    volume_data = torch.zeros((3, 3, 3), device=device)
    volume_data[1, 1, 1] = 1.
    transformation = Transformation(rotation=torch.zeros(3), translation=torch.tensor([0., 0., 1.5]))
    voxel_spacing = torch.tensor([1., 1., 1.])
    detector_spacing = torch.tensor([1., 1.])
    scene_geometry = SceneGeometry(source_distance=3.)
    output_size = torch.Size([3, 3])
    ret = generate_drr_python(volume_data, transformation=transformation, voxel_spacing=voxel_spacing,
                              detector_spacing=detector_spacing, scene_geometry=scene_geometry, output_size=output_size)
    assert isinstance(ret, torch.Tensor)
    assert ret.size() == output_size
    assert ret.device == device


def test_generate_drr():
    device = torch.device('cpu')

    volume_data = torch.full((5, 5, 5), -1000, device=device, dtype=torch.float32)
    volume_data[1:4, 1:4, 1:4] = 0
    volume_data[1:4, 1, 1:4] = 500
    volume_data[2, 2, 2] = 1000
    volume_data[1:4, 3, 3] = 2000
    # density = diffdrr.data.transform_hu_to_density(volume_data, bone_attenuation_multiplier=1.0)
    density = volume_data
    density[density < -800.0] = -800.0
    density -= density.min()
    density /= density.max()

    transformation = Transformation(rotation=torch.zeros(3), translation=torch.tensor([0., 0., 1.5]))
    voxel_spacing = torch.tensor([1., 1., 1.])
    detector_spacing = torch.tensor([1., 1.])
    scene_geometry = SceneGeometry(source_distance=3.)
    output_size = torch.Size([5, 5])

    drr_python = generate_drr_python(density, transformation=transformation, voxel_spacing=voxel_spacing,
                                     detector_spacing=detector_spacing, scene_geometry=scene_geometry,
                                     output_size=output_size)

    drr_diffdrr = generate_drr(density, transformation=transformation, voxel_spacing=voxel_spacing,
                               detector_spacing=detector_spacing, scene_geometry=scene_geometry,
                               output_size=output_size)

    # # Plotting DRR
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(drr_python.cpu())
    # axes.axis('square')
    # axes.set_title("g")
    # axes.set_xlabel("x")
    # axes.set_ylabel("y")
    # plt.colorbar(mesh)
    #
    # # Plotting DRR
    # _, axes = plt.subplots()
    # mesh = axes.pcolormesh(drr_diffdrr.cpu())
    # axes.axis('square')
    # axes.set_title("g")
    # axes.set_xlabel("x")
    # axes.set_ylabel("y")
    # plt.colorbar(mesh)
    #
    # plt.show()

    assert drr_python == pytest.approx(drr_diffdrr, abs=0.002)


def test_plane_integrals():
    device = torch.device('cpu')
    volume_data = torch.zeros((3, 3, 3), device=device)
    volume_data[1, 1, 1] = 1.
    voxel_spacing = torch.tensor([1., 1., 1.])
    phi_values = torch.tensor([0.1])
    # phi_values = torch.tensor([torch.pi * .25])
    theta_values = torch.tensor([0.1])
    # theta_values = torch.tensor([torch.pi * .25])
    r_values = torch.tensor([0.])
    from_extension = ExtensionTest.radon3d(volume_data, voxel_spacing, phi_values, theta_values, r_values,
                                           samples_per_direction=10)
    phi_values, theta_values, r_values = torch.meshgrid(phi_values, theta_values, r_values)
    from_python = plane_integrals(volume_data, voxel_spacing=voxel_spacing, phi_values=phi_values,
                                  theta_values=theta_values, r_values=r_values, samples_per_direction=10)
    assert from_extension.item() == pytest.approx(from_python.item())

    volume_data = torch.zeros((10, 10, 10), device=device)
    volume_data[0, :, :] = 1.
    voxel_spacing = torch.tensor([1., 1., 1.])
    volume_size = (torch.tensor(volume_data.size(), dtype=torch.float32) * voxel_spacing).square().sum().sqrt()
    phi_values = torch.linspace(-.5 * torch.pi, .5 * torch.pi, 4, device=device)
    theta_values = torch.linspace(-.5 * torch.pi, .5 * torch.pi, 4, device=device)
    r_values = torch.linspace(-.5 * volume_size, .5 * volume_size, 4, device=device)
    phi_values, theta_values, r_values = torch.meshgrid(phi_values, theta_values, r_values)
    radon = ExtensionTest.radon3d(volume_data, voxel_spacing, phi_values, theta_values, r_values,
                                  samples_per_direction=500)
    radon_python = plane_integrals(volume_data, voxel_spacing=voxel_spacing, phi_values=phi_values,
                                   theta_values=theta_values, r_values=r_values, samples_per_direction=500)
    assert (radon_python - radon).abs().mean() / (
            .5 * (radon_python.max() - radon_python.min() + radon.max() - radon.min())) == pytest.approx(0., abs=1e-4)
