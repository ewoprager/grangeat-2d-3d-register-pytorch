import torch

__all__ = ["torch_polyfit", "fit_power_relationship"]


def torch_polyfit(xs: torch.Tensor, ys: torch.Tensor, *, degree: int = 1) -> torch.Tensor:
    assert xs.size() == ys.size()

    A = torch.stack([xs.pow(i) for i in range(degree + 1)], dim=1)  # size = (n, degree + 1)
    B = ys.unsqueeze(1)

    ret = torch.linalg.lstsq(A, B)

    return ret.solution


def fit_power_relationship(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """
    y = a * x^b
    :param xs:
    :param ys:
    :return: (2,) tensor containing [a, b]
    """
    assert xs.size() == ys.size()
    assert (xs > 0.0).all()
    assert (ys > 0.0).all()

    ln_xs = xs.log()
    ln_ys = ys.log()
    coefficients = torch_polyfit(ln_xs, ln_ys, degree=1)
    return torch.tensor([coefficients[0].exp(), coefficients[1]])
