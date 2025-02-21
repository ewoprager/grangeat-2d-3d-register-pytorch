import pytest
import torch

import registration.objective_function as objective_function
import Extension


def test_zncc():
    # a = torch.tensor([1.56, 2.35, 3.361])
    # b = torch.tensor([1.7, 2.66, 3.11])
    a = torch.rand((100, 100))
    b = torch.rand((100, 100))
    vanilla_cpu = objective_function.zncc(a, b)
    vanilla_cuda = objective_function.zncc(a.cuda(), b.cuda())
    cpu = Extension.normalised_cross_correlation(a, b)
    cuda = Extension.normalised_cross_correlation(a.cuda(), b.cuda())
    assert vanilla_cuda.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)
    assert cpu.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)
    assert cuda.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)

    b *= -1.
    vanilla_cpu = objective_function.zncc(a, b)
    vanilla_cuda = objective_function.zncc(a.cuda(), b.cuda())
    cpu = Extension.normalised_cross_correlation(a, b)
    cuda = Extension.normalised_cross_correlation(a.cuda(), b.cuda())
    assert vanilla_cuda.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)
    assert cpu.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)
    assert cuda.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)

    b = torch.zeros_like(a)
    vanilla_cpu = objective_function.zncc(a, b)
    vanilla_cuda = objective_function.zncc(a.cuda(), b.cuda())
    cpu = Extension.normalised_cross_correlation(a, b)
    cuda = Extension.normalised_cross_correlation(a.cuda(), b.cuda())
    assert vanilla_cuda.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)
    assert cpu.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)
    assert cuda.item() == pytest.approx(vanilla_cpu.item(), abs=1e-4)
