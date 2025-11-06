#pragma once

#include "Common.h"

namespace reg23 {

/**
 * @ingroup pytorch_functions
 * @brief Additionally returns intermediate quantities useful for evaluating the backward pass
 * @param a a tensor of any size containing `torch.float32`s
 * @param b a tensor of the same size as `a` containing `torch.float32`s
 * @return  @return a tuple: (ZNCC; sum of values in `a`; sum of values in `b`; the numerator of the fraction shown below; the left sqrt in the denominator of the fraction shown below; the right sqrt in the denominator of the fraction shown below)
 *
 * ZNCC = (N * sum(a_i * b_i) - sum(a_i) * sum(b_i)) / (sqrt(N * sum(a_i^2) - sum(a_i)^2) * sqrt(N * sum(b_i^2) - sum(b_i)^2))
 */
std::tuple<at::Tensor, double, double, double, double, double> NormalisedCrossCorrelation_CPU(
	const at::Tensor &a, const at::Tensor &b);

/**
 * @ingroup pytorch_functions
 * @brief An implementation of reg23::NormalisedCrossCorrelation_CPU that uses CUDA parallelisation
 */
__host__ std::tuple<at::Tensor, double, double, double, double, double> NormalisedCrossCorrelation_CUDA(
	const at::Tensor &a, const at::Tensor &b);

struct Similarity {

	__host__ static void Common(const at::Tensor &a, const at::Tensor &b, at::DeviceType device) {
		// a and b should contain floats or doubles, match in size and be on the chosen device
		TORCH_CHECK(a.sizes() == b.sizes())
		TORCH_CHECK(a.dtype() == b.dtype())
		TORCH_CHECK(a.dtype() == at::kFloat)
		TORCH_INTERNAL_ASSERT(a.device().type() == device)
		TORCH_INTERNAL_ASSERT(b.device().type() == device)
	}
};

} // namespace reg23