import pytest
import torch

from registration.lib.structs import Transformation

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
