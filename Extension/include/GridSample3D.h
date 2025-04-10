#pragma once

#include "Common.h"
#include "Vec.h"
#include "Texture.h"

namespace ExtensionTest {

at::Tensor GridSample3D_CPU(const at::Tensor &input, const at::Tensor &grid);

__host__ at::Tensor GridSample3D_CUDA(const at::Tensor &input, const at::Tensor &grid);

template <typename texture_t> struct GridSample3D {

	struct CommonData {
		texture_t inputTexture{};
		at::Tensor flatOutput{};
	};

	__host__ static CommonData Common(const at::Tensor &input, const at::Tensor &grid, at::DeviceType device) {
		// input should be a 3D tensor of floats on the chosen device
		TORCH_CHECK(input.sizes().size() == 3)
		TORCH_CHECK(input.dtype() == at::kFloat)
		TORCH_INTERNAL_ASSERT(input.device().type() == device)
		// grid should be a tensor of floats or doubles with a final dimension of 3 on the chosen device
		TORCH_CHECK(grid.sizes().back() == 3);
		TORCH_CHECK(grid.dtype() == at::kFloat || grid.dtype() == at::kDouble);
		TORCH_INTERNAL_ASSERT(grid.device().type() == device)

		CommonData ret{};
		const at::Tensor inputContiguous = input.contiguous();
		const float *inputPtr = inputContiguous.data_ptr<float>();
		const Vec<int64_t, 3> inputSize = Vec<int64_t, 3>::FromIntArrayRef(input.sizes()).Flipped();
		const Vec<double, 3> inputSpacing = 2.0 / (inputSize - static_cast<int64_t>(1)).StaticCast<double>();
		ret.inputTexture = texture_t{inputPtr, inputSize, inputSpacing, Vec<double, 3>::Full(0.0),
		                             {TextureAddressMode::ZERO, TextureAddressMode::ZERO, TextureAddressMode::WRAP}};
		ret.flatOutput = torch::zeros(at::IntArrayRef({grid.numel() / 3}), inputContiguous.options());
		return ret;
	}
};

} // namespace ExtensionTest