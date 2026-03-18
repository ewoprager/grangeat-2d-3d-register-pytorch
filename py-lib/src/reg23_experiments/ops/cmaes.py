from typing import Callable
from math import log, floor, sqrt, exp

import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class CMAES:
    """
    Translated directly from the Matlab code here: https://en.wikipedia.org/wiki/CMA-ES

    """

    def __init__(self, dimensionality: int, objective_function: Callable[[torch.Tensor], torch.Tensor],
                 initial_position: torch.Tensor):
        self._dimensionality = dimensionality
        self._objective_function = objective_function

        self._sigma: float = 0.3
        self._lambda: int = 4 + int(floor(3.0 * log(float(self._dimensionality))))
        self._mu: int = int(floor(0.5 * float(self._lambda)))
        self._weights: torch.Tensor = log(float(self._mu) + 0.5) - torch.range(1, self._mu).log()
        self._weights /= self._weights.sum()  # size = (mu,)
        self._mu_eff: float = (self._weights.sum().square() / self._weights.square().sum()).item()

        self._cc: float = (4.0 + self._mu_eff / float(self._dimensionality)) / (
                float(self._dimensionality + 4) + 2.0 * self._mu_eff / float(self._dimensionality))
        self._cs: float = (self._mu_eff + 2.0) / (float(self._dimensionality) + self._mu_eff + 5.0)
        self._c1: float = 2.0 / ((float(self._dimensionality) + 1.3) ** 2 + self._mu_eff)
        self._cmu: float = min(1.0 - self._c1, 2.0 * (self._mu_eff - 2.0 + 1.0 / self._mu_eff) / (
                float(self._dimensionality + 2) ** 2 + self._mu_eff))
        self._damps: float = 1.0 + 2.0 * max(0.0, sqrt(
            (self._mu_eff - 1.0) / float(self._dimensionality + 1)) - 1.0) + self._cs

        self._mean = initial_position
        self._isotropic_evo_path = torch.zeros_like(self._mean)
        self._anisotropic_evo_path = torch.zeros_like(self._mean)
        self._b_matrix = torch.eye(self._dimensionality)
        self._d_diagonal = torch.ones_like(self._mean)
        # the covariance matrix is B * diag(D^2) * B^T
        self._covariance_matrix = self._b_matrix @ torch.diag(self._d_diagonal.square()) @ self._b_matrix.t()
        # the inverse sqrt covariance matrix is B * diag(D^-1) * B^T
        self._inverse_sqrt_c_matrix = self._b_matrix @ torch.diag(1.0 / self._d_diagonal) @ self._b_matrix.t()
        self._eigeneval: int = 0
        self._chi_n: float = sqrt(float(self._dimensionality)) * (1.0 - 1.0 / float(4 * self._dimensionality) + 1.0 / (
                21.0 * float(self._dimensionality * self._dimensionality)))
        self._eval_count: int = 0

    def iterate(self):
        agents = torch.empty((self._lambda, self._dimensionality + 1))  # | <- x -> | f(x) |
        # take lambda samples according to the current distribution
        for i in range(self._lambda):
            agents[i, :self._dimensionality] = MultivariateNormal(self._mean,
                                                                  self._sigma * self._covariance_matrix).sample()
            agents[i, -1] = self._objective_function(agents[i, 0:self._dimensionality])
        self._eval_count += self._lambda
        # sort the samples by their o.f. values
        _, indices_sorted_by_f = torch.sort(agents[:, -1])
        agents = agents[indices_sorted_by_f]  # size = (lambda, dimensionality + 1)
        old_mean = self._mean.clone()
        # update the mean to be a weighted sum of the mu best samples
        self._mean = torch.einsum('ij,i->j', agents[:self._mu, :self._dimensionality], self._weights)
        mean_delta = self._mean - old_mean
        # update the isotropic evolution path
        self._isotropic_evo_path = (1.0 - self._cs) * self._isotropic_evo_path + sqrt(
            1.0 - (1.0 - self._cs) * (1.0 - self._cs)) * sqrt(
            self._mu_eff) * self._inverse_sqrt_c_matrix @ mean_delta / self._sigma
        # update the anisotropic evolution path
        h_sig: float = float(torch.linalg.vector_norm(self._isotropic_evo_path) / sqrt(
            1.0 - (1.0 - self._cs) ** (float(2 * self._eval_count) / self._lambda)) / self._chi_n < 1.4) + 2.0 / float(
            self._dimensionality + 1)
        self._anisotropic_evo_path = (1.0 - self._cc) * self._anisotropic_evo_path + h_sig * sqrt(
            self._cc * (2.0 - self._cc) * self._mu_eff) * mean_delta / self._sigma
        # update the covariance matrix
        ar_tmp = (1.0 / self._sigma) * (agents[:self._mu, :self._dimensionality] - old_mean.unsqueeze(0))
        self._covariance_matrix = (  #
                (1.0 - self._c1 - self._cmu) * self._covariance_matrix +  #
                self._c1 * (torch.outer(self._anisotropic_evo_path, self._anisotropic_evo_path)  #
                            + (1.0 - h_sig) * self._cc * (2.0 - self._cc) * self._covariance_matrix)  #
                + self._cmu * ar_tmp @ torch.diag(self._weights) @ ar_tmp.t()  #
        )
        # update sigma
        self._sigma *= exp(
            (self._cs / self._damps) * (torch.linalg.vector_norm(self._isotropic_evo_path).item() / self._chi_n - 1.0))
        if self._eval_count - self._eigeneval > self._lambda / (self._c1 + self._cmu) / float(
                self._dimensionality) / 10.0:
            self._eigeneval = self._eval_count
            L = torch.tril(self._covariance_matrix)
            diag = torch.diagonal(L)
            L = L - torch.diag(diag) + torch.diag(torch.exp(diag))
            self._covariance_matrix = L @ L.t()  # enforce symmetry
            self._d_diagonal, self._b_matrix = torch.linalg.eigh(self._covariance_matrix)
            self._d_diagonal = self._d_diagonal.sqrt()
            self._inverse_sqrt_c_matrix = self._b_matrix @ torch.diag(1.0 / self._d_diagonal) @ self._b_matrix.t()
