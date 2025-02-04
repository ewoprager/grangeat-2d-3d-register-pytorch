import pytest

from registration.geometry import *


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
    ret = fixed_polar_to_moving_cartesian(input_grid, scene_geometry=scene_geometry, transformation=transformation)
    assert isinstance(ret, torch.Tensor)
    assert ret.device == input_grid.phi.device
    assert ret.size() == torch.Size([1, 3])
    assert ret[0, 0].item() == pytest.approx((b * sin_alpha * cos_alpha).item(), abs=1e-4)
    assert ret[0, 1].item() == pytest.approx(0., abs=1e-4)
    assert ret[0, 2].item() == pytest.approx(b * sin_alpha.square().item(), abs=1e-4)

    transformation = Transformation(rotation=torch.zeros(3, device=device),
                                    translation=torch.tensor([b * sin_alpha * cos_alpha, 0., b * sin_alpha.square()]))
    ret = fixed_polar_to_moving_cartesian(input_grid, scene_geometry=scene_geometry, transformation=transformation)
    assert ret[0, 0].item() == pytest.approx(0., abs=1e-4)
    assert ret[0, 1].item() == pytest.approx(0., abs=1e-4)
    assert ret[0, 2].item() == pytest.approx(0., abs=1e-4)

    angle = 0.32341
    input_grid = Sinogram2dGrid(phi=torch.tensor([angle], device=device), r=-a)
    transformation = Transformation(rotation=torch.zeros(3, device=device), translation=torch.zeros(3, device=device))
    ret_a = fixed_polar_to_moving_cartesian(input_grid, scene_geometry=scene_geometry, transformation=transformation)
    input_grid = Sinogram2dGrid(phi=torch.tensor([angle + torch.pi], device=device), r=a)
    ret_b = fixed_polar_to_moving_cartesian(input_grid, scene_geometry=scene_geometry, transformation=transformation)
    assert ret_a[0, 0].item() == pytest.approx(ret_b[0, 0].item(), abs=1e-4)
    assert ret_a[0, 1].item() == pytest.approx(ret_b[0, 1].item(), abs=1e-4)
    assert ret_a[0, 2].item() == pytest.approx(ret_b[0, 2].item(), abs=1e-4)

    input_grid = Sinogram2dGrid(phi=torch.tensor([.5 * torch.pi], device=device), r=a)
    ret = fixed_polar_to_moving_cartesian(input_grid, scene_geometry=scene_geometry, transformation=transformation)
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
    assert ret.phi.item() == pytest.approx(.5 * torch.pi, abs=1e-4)
    assert ret.theta.item() == pytest.approx(0., abs=1e-4)
    assert ret.r.item() == pytest.approx(1., abs=1e-4)

    ret = moving_cartesian_to_moving_spherical(torch.tensor([0., 0., 1.], device=device))
    assert ret.theta.item() == pytest.approx(.5 * torch.pi, abs=1e-4)
    assert ret.r.item() == pytest.approx(1., abs=1e-4)

    ret = moving_cartesian_to_moving_spherical(torch.tensor([-1., 0., 0.], device=device))
    assert ret.phi.item() == pytest.approx(0., abs=1e-4)
    assert ret.theta.item() == pytest.approx(0., abs=1e-4)
    assert ret.r.item() == pytest.approx(-1., abs=1e-4)


def test_generate_drr():
    device = torch.device('cpu')
    volume_data = torch.zeros((3, 3, 3), device=device)
    volume_data[1, 1, 1] = 1.
    transformation = Transformation(rotation=torch.zeros(3), translation=torch.tensor([0., 0., 1.5]))
    voxel_spacing = torch.tensor([1., 1., 1.])
    detector_spacing = torch.tensor([1., 1.])
    scene_geometry = SceneGeometry(source_distance=3.)
    output_size = torch.Size([3, 3])
    ret = generate_drr(volume_data, transformation=transformation, voxel_spacing=voxel_spacing,
                 detector_spacing=detector_spacing, scene_geometry=scene_geometry, output_size=output_size)
    assert isinstance(ret, torch.Tensor)
    assert ret.size() == output_size
    assert ret.device == device

