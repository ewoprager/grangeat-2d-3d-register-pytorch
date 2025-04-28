#pragma once

#include "Common.h"

namespace ExtensionTest {

/**
 * @param a A tensor of `torch.float32`s of any size
 * @param b A tensor of `torch.float32`s matching `a` in size
 * @return The zero-normalised cross-correlation between the given tensors.
 */
at::Tensor NormalisedCrossCorrelation_CPU(const at::Tensor &a, const at::Tensor &b);

/**
 * An implementation of `NormalisedCrossCorrelation_CPU` that uses CUDA parallisation.
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

} // namespace ExtensionTest