#pragma once

#include "Common.h"
#include "Texture.h"
#include "Vec.h"

namespace ExtensionTest {

/**
 * @brief Sample the given 3D input tensor at the positions given in grid according to the given address mode using
 * bilinear interpolation. This implementation is single-threaded.
 * @param input A vector of `torch.float32`s; must have 3 dimensions
 * @param grid A vector of `torch.float32`s; can have any number of dimensions, but the last dimension must be 3 -
 * representing 3D locations within the `input` volume between (-1, -1, -1) and (1, 1, 1)
 * @param addressMode Either "wrap" or "zero"
 * @return A vector of `torch.float32`s  with the same size as `grid`, minus the last dimension.
 *
 * # Address modes
 * - "zero": sampling locations outside (-1, -1, -1) and (1, 1, 1) will be read as 0
 * - "wrap": sampling locations (x, y, z) outside (-1, -1, -1) and (1, 1, 1) will be read wrapped back according to ((x
 * + 1) mod 2 - 1, (y + 1) mod 2 - 1, (y + 1) mod 2 - 1)
 */
at::Tensor GridSample3D_CPU(const at::Tensor &input, const at::Tensor &grid, const std::string &addressMode);

/**
 * @brief Sample the given 3D input tensor at the positions given in grid according to the given address mode using
 * bilinear interpolation. This implementation uses CUDA parallelisation.
 * @param input A vector of `torch.float32`s; must have 3 dimensions
 * @param grid A vector of `torch.float32`s; can have any number of dimensions, but the last dimension must be 3 -
 * representing 3D locations within the `input` volume between (-1, -1, -1) and (1, 1, 1)
 * @param addressMode Either "wrap" or "zero"
 * @return A vector of `torch.float32`s  with the same size as `grid`, minus the last dimension.
 *
 * # Address modes
 * - "zero": sampling locations outside (-1, -1, -1) and (1, 1, 1) will be read as 0
 * - "wrap": sampling locations (x, y, z) outside (-1, -1, -1) and (1, 1, 1) will be read wrapped back according to ((x
 * + 1) mod 2 - 1, (y + 1) mod 2 - 1, (y + 1) mod 2 - 1)
 */
__host__ at::Tensor GridSample3D_CUDA(const at::Tensor &input, const at::Tensor &grid, const std::string &addressMode);

/**
 * @tparam texture_t Type of the texture object that input data will be converted to for sampling.
 *
 * This struct is used as a namespace for code that is shared between different implementations of GridSample3D_...
 * functions
 */
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
		// grid should be a tensor of floats with a final dimension of 3 on the chosen device
		TORCH_CHECK(grid.sizes().back() == 3);
		TORCH_CHECK(grid.dtype() == at::kFloat);
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