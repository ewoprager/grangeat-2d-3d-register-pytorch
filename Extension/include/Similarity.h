#pragma once

#include "Common.h"

namespace reg23 {

/**
 * @ingroup pytorch_functions
 * @param a a tensor of any size containing `torch.float32`s
 * @param b a tensor of the same size as `a` containing `torch.float32`s
 * @return The zero-normalised cross-correlation between the given tensors.
 */
at::Tensor NormalisedCrossCorrelation_CPU(const at::Tensor &a, const at::Tensor &b);

/**
 * @ingroup pytorch_functions
 * @brief An implementation of reg23::NormalisedCrossCorrelation_CPU that uses CUDA parallisation
 */
__host__ at::Tensor NormalisedCrossCorrelation_CUDA(const at::Tensor &a, const at::Tensor &b);

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