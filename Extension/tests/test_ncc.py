import pytest
import torch

from Extension import normalised_cross_correlation, autograd


def test_ncc():
    a = torch.rand((10, 10))
    b = torch.rand(a.size())

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device_name in devices:
        device = torch.device(device_name)
        a = a.detach().to(device=device)
        b = b.to(device=device)
        zncc_basic = normalised_cross_correlation(a, b)
        a.requires_grad = True
        zncc = autograd.normalised_cross_correlation(a, b)
        assert zncc.clone().detach().cpu() == pytest.approx(zncc_basic.cpu(), rel=1.0e-4, abs=1.0e-5)
        zncc.backward()
        out = torch.empty_like(a)
        epsilon = 1.0e-4
        for j in range(out.size(0)):
            for i in range(out.size(1)):
                a_delta = a.clone().detach()
                a_delta[j, i] += epsilon
                out[j, i] = (autograd.normalised_cross_correlation(a_delta, b) - zncc) / epsilon
        assert out.detach().cpu() == pytest.approx(a.grad.detach().cpu(), abs=1.0e-3, rel=1.0e-2)
