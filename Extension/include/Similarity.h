#pragma once

#include "Common.h"

namespace ExtensionTest {

at::Tensor NormalisedCrossCorrelation_CPU(const at::Tensor &a, const at::Tensor &b);

__host__ at::Tensor NormalisedCrossCorrelation_CUDA(const at::Tensor &a, const at::Tensor &b);

struct Similarity {

	__host__ static void Common(const at::Tensor &a, const at::Tensor &b, at::DeviceType device) {
		// a and b should contain floats or doubles, match in size and be on the chosen device
		TORCH_CHECK(a.sizes() == b.sizes())
		TORCH_CHECK(a.dtype() == at::kFloat || a.dtype() == at::kDouble)
		TORCH_CHECK(b.dtype() == at::kFloat || b.dtype() == at::kDouble)
		TORCH_INTERNAL_ASSERT(a.device().type() == device)
		TORCH_INTERNAL_ASSERT(b.device().type() == device)
	}

};

} // namespace ExtensionTest