import pytest
import torch

from reg23_experiments.data.structs import Transformation
from reg23_experiments.ops.autograd_impl import TransformationToMatrix


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

    # t2 = t.vectorised().detach().clone()
    # t2.requires_grad = True
    # torch.autograd.gradcheck(TransformationToMatrix.apply, t2)
    #
    # return

    print("Kornia delta:")
    epsilon = 1.0e-4
    t_delta = t.clone()
    t_delta.rotation[0] += epsilon
    h_kornia_delta = t_delta.get_h()
    print((h_kornia_delta.sum() - h_kornia.sum()) / epsilon)

    print("Custom delta:")
    epsilon = 1.0e-4
    t2_delta = t.vectorised().clone()
    t2_delta[0] += epsilon
    h_custom_delta = TransformationToMatrix.apply(t2_delta)
    print((h_custom_delta.sum() - h_custom.sum()) / epsilon)

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
