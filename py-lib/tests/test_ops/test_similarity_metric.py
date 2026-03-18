import pytest
import torch

from reg23_experiments.ops import similarity_metric
import reg23


def test_ncc():
    cuda_available = torch.cuda.is_available()

    # a = torch.tensor([1.56, 2.35, 3.361])
    # b = torch.tensor([1.7, 2.66, 3.11])
    a = torch.rand((100, 100), dtype=torch.float32)
    b = torch.rand((100, 100), dtype=torch.float32)
    vanilla_cpu = similarity_metric.ncc(a, b)
    cpu = reg23.normalised_cross_correlation(a, b)
    assert cpu.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)
    if cuda_available:
        vanilla_cuda = similarity_metric.ncc(a.cuda(), b.cuda())
        cuda = reg23.normalised_cross_correlation(a.cuda(), b.cuda())
        assert vanilla_cuda.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)
        assert cuda.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)

    b *= -1.
    vanilla_cpu = similarity_metric.ncc(a, b)
    cpu = reg23.normalised_cross_correlation(a, b)
    assert cpu.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)
    if cuda_available:
        vanilla_cuda = similarity_metric.ncc(a.cuda(), b.cuda())
        cuda = reg23.normalised_cross_correlation(a.cuda(), b.cuda())
        assert vanilla_cuda.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)
        assert cuda.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)

    b = torch.ones_like(a)
    vanilla_cpu = similarity_metric.ncc(a, b)
    cpu = reg23.normalised_cross_correlation(a, b)
    assert cpu.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)
    if cuda_available:
        vanilla_cuda = similarity_metric.ncc(a.cuda(), b.cuda())
        cuda = reg23.normalised_cross_correlation(a.cuda(), b.cuda())
        assert vanilla_cuda.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)
        assert cuda.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)


def test_local_ncc():
    # cuda_available = torch.cuda.is_available()

    a = torch.rand((4, 4))
    b = a.clone()
    b[0:2, 2:4] *= -1.0
    b[2:4, 0:2] = 1.0
    res = similarity_metric.local_ncc(a, b, kernel_size=2)
    assert res.size() == torch.Size([])
    assert res.item() == pytest.approx(0.25 * (1.0 + -1.0 + 0.0 + 1.0))


def test_gradient_correlation():
    # cuda_available = torch.cuda.is_available()

    a = torch.rand((10, 10))
    b = torch.rand((10, 10))
    res = similarity_metric.gradient_correlation(a, b, gradient_method="sobel")
    res = similarity_metric.gradient_correlation(a, b, gradient_method="central_difference")

    b = torch.rand((100,))
    with pytest.raises(AssertionError):
        res = similarity_metric.gradient_correlation(a, b)
