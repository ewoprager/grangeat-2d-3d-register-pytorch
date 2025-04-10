#pragma once

#include "Common.h"
#include "Vec.h"
#include "Texture.h"

namespace ExtensionTest {

at::Tensor GridSample3D_CPU(const at::Tensor &input, const at::Tensor &grid, const std::string &addressMode);

__host__ at::Tensor GridSample3D_CUDA(const at::Tensor &input, const at::Tensor &grid, const std::string &addressMode);

template <typename texture_t> struct GridSample3D {

	struct CommonData {
		texture_t inputTexture{};
		at::Tensor flatOutput{};
	};

	__host__ static CommonData Common(const at::Tensor &input, const at::Tensor &grid, const std::string &addressMode,
	                                  at::DeviceType device) {
		// input should be a 3D tensor of floats on the chosen device
		TORCH_CHECK(input.sizes().size() == 3)
		TORCH_CHECK(input.dtype() == at::kFloat)
		TORCH_INTERNAL_ASSERT(input.device().type() == device)
		// grid should be a tensor of doubles with a final dimension of 3 on the chosen device
		TORCH_CHECK(grid.sizes().back() == 3);
		TORCH_CHECK(grid.dtype() == at::kDouble);
		TORCH_INTERNAL_ASSERT(grid.device().type() == device)
		// addressMode should be one of the valid values:
		TextureAddressMode am = TextureAddressMode::ZERO;
		if (addressMode == "wrap") {
			am = TextureAddressMode::WRAP;
		} else if (addressMode != "zero") {
			TORCH_WARN(
				"Invalid address mode string given. Valid values are: 'zero', 'wrap'. Using default value: 'zero'.")
		}

		CommonData ret{};
		const at::Tensor inputContiguous = input.contiguous();
		const float *inputPtr = inputContiguous.data_ptr<float>();
		const Vec<int64_t, 3> inputSize = Vec<int64_t, 3>::FromIntArrayRef(input.sizes()).Flipped();
		const Vec<double, 3> inputSpacing = 2.0 / (inputSize - static_cast<int64_t>(1)).StaticCast<double>();
		ret.inputTexture = texture_t{inputPtr, inputSize, inputSpacing, Vec<double, 3>::Full(0.0),
		                             texture_t::AddressModeType::Full(am)};
		ret.flatOutput = torch::zeros(at::IntArrayRef({grid.numel() / 3}), inputContiguous.options());
		return ret;
	}
};

} // namespace ExtensionTest