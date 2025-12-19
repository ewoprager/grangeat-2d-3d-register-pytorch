import torch

__all__ = ["TransformationToMatrix"]


class TransformationToMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transformation: torch.Tensor) -> torch.Tensor:
        assert transformation.size() == torch.Size([6])

        theta = torch.linalg.vector_norm(transformation[0:3])
        cos_theta = theta.cos()
        sin_theta = theta.sin()
        u = transformation[0:3] / (theta + 1.0e-10)
        k = torch.tensor([  #
            [0.0, u[2], -u[1]],  #
            [-u[2], 0.0, u[0]],  #
            [u[1], -u[0], 0.0],  #
        ])
        r = cos_theta * torch.eye(3, device=transformation.device, dtype=transformation.dtype) + (
                1.0 - cos_theta) * u.outer(u) + sin_theta * k

        ctx.save_for_backward(transformation, theta, cos_theta, sin_theta, u, k)

        return torch.cat(
            (torch.cat((r, transformation[3:6].unsqueeze(0)), dim=0), torch.tensor([[0.0], [0.0], [0.0], [1.0]])),
            dim=1)

    @staticmethod
    def backward(ctx, d_loss_d_matrix):
        """
        :param ctx:
        :param d_loss_d_matrix: size = (4, 4)
        :return:
        """
        (transformation, theta, cos_theta, sin_theta, u, k) = ctx.saved_tensors

        d_r_d_theta = sin_theta * (-torch.eye(3, device=transformation.device, dtype=transformation.dtype) + u.outer(
            u)) + cos_theta * k  # size = (3, 3)

        d_theta_d_axis_angle = u  # size (3,): (components of axis angle)

        d_uut_d_ux = torch.cat((u.unsqueeze(0), torch.zeros(2, 3)), dim=0)
        d_uut_d_ux += d_uut_d_ux.clone().t()

        d_uut_d_uy = torch.cat((torch.zeros(1, 3), u.unsqueeze(0), torch.zeros(1, 3)), dim=0)
        d_uut_d_uy += d_uut_d_uy.clone().t()

        d_uut_d_uz = torch.cat((torch.zeros(2, 3), u.unsqueeze(0)), dim=0)
        d_uut_d_uz += d_uut_d_uz.clone().t()

        d_r_d_ux = (1.0 - cos_theta) * d_uut_d_ux + torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -sin_theta], [0.0, sin_theta, 0.0]])  # size = (3, 3)
        d_r_d_uy = (1.0 - cos_theta) * d_uut_d_uy + torch.tensor(
            [[0.0, 0.0, sin_theta], [0.0, 0.0, 0.0], [-sin_theta, 0.0, 0.0]])  # size = (3, 3)
        d_r_d_uz = (1.0 - cos_theta) * d_uut_d_uz + torch.tensor(
            [[0.0, -sin_theta, 0.0], [sin_theta, 0.0, 0.0], [0.0, 0.0, 0.0]])  # size = (3, 3)
        d_r_d_u = torch.stack((d_r_d_ux, d_r_d_uy, d_r_d_uz), dim=0)  # size = (3, 3, 3): (components of u, 3x3 r)

        d_u_d_axis_angle = (theta - u.outer(
            transformation[0:3])) / theta.square()  # size = (3, 3): (components of axis angle, components of u)

        d_r_d_axis_angle = torch.einsum(  #
            "ji,k->kji", d_r_d_theta, d_theta_d_axis_angle) + torch.einsum(  #
            "lji,kl->kji", d_r_d_u, d_u_d_axis_angle)  # size = (3, 3, 3): (components of axis angle, 3x3 matrix)

        d_matrix_d_axis_angle = torch.cat((  #
            torch.cat((d_r_d_axis_angle, torch.zeros((3, 3, 1))), dim=2),  #
            torch.zeros((3, 1, 4))  #
        ), dim=1)

        d_matrix_d_translation = torch.zeros((3, 4, 4))
        d_matrix_d_translation[0, 3, 0] = 1.0
        d_matrix_d_translation[1, 3, 1] = 1.0
        d_matrix_d_translation[2, 3, 2] = 1.0

        d_matrix_d_transformation = torch.cat((  #
            d_matrix_d_axis_angle,  #
            d_matrix_d_translation  #
        ), dim=0)  # size = (6, 4, 4)

        d_loss_d_transformation = torch.einsum("ji,kji->k", d_loss_d_matrix, d_matrix_d_transformation)  # size = (6,)

        return d_loss_d_transformation
