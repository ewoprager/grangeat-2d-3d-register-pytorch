import pytest
import torch

from registration.lib.structs import Transformation
from registration.lib.autograd_impl import TransformationToMatrix


# def test_transformation_matrix_autograd():
#     t = Transformation(rotation=torch.tensor([1.0, 2.0, 3.0]), translation=torch.tensor([3.0, 2.0, 1.0]))
#     t.rotation.requires_grad = True
#     t.translation.requires_grad = True
#     h = t.get_h()
#     print(h)
#
#     grad_matrix = torch.zeros_like(h)
#     grad_matrix[0, 0] = 1.0
#
#     h.backward(grad_matrix)
#     print(t.rotation.grad)

def test_transformation_matrix_custom():
    print("")
    t = Transformation.random_uniform()
    h_kornia = t.get_h()
    print(h_kornia)
    h_custom = TransformationToMatrix.apply(t.vectorised())
    print(h_custom)

    print("Kornia delta:")
    epsilon = 1.0e-4
    t.rotation[0] += epsilon
    h_kornia_delta = t.get_h()
    print((h_kornia_delta.sum() - h_kornia.sum()) / epsilon)

    print("Kornia:")

    t.rotation.requires_grad = True
    t.translation.requires_grad = True
    h_kornia = t.get_h()
    h_kornia.backward(torch.ones_like(h_kornia))
    print(t.rotation.grad)
    print(t.translation.grad)

    print("Custom:")

    t2 = t.vectorised().detach().clone()
    t2.requires_grad = True
    h_custom = TransformationToMatrix.apply(t2)
    h_custom.backward(torch.ones_like(h_custom))
    print(t2.grad)
