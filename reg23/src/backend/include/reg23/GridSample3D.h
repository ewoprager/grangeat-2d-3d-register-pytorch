/**
 * @file
 * @brief Implementations of grid sampling functions
 */

#pragma once

#include "Common.h"
#include "Texture.h"

namespace reg23 {

/**
 * @ingroup pytorch_functions
 * @brief Sample the given 3D input tensor at the positions given in grid according to the given address mode using
 * bilinear interpolation. This implementation is single-threaded.
 * @param input a tensor of size (P,Q,R) containing `torch.float32`s: The volume to sample
 * @param grid a tensor of size (..., 3) containing `torch.float32`s: The grid to sample at; the last dimension represents 3D locations within the `input` volume between (-1, -1, -1) and (1, 1, 1)
 * @param addressModeX either "wrap" or "zero": The address mode used for dimension X
 * @param addressModeY either "wrap" or "zero": The address mode used for dimension Y
 * @param addressModeZ either "wrap" or "zero": The address mode used for dimension Z
 * @return a tensor of size (...) containing `torch.float32`s: The sampled values in the given volume at the locations in the given grid; this will be the same size as `grid`, minus the last dimension.
 *
 * # Address modes
 * - "zero": sampling locations outside (-1, -1, -1) and (1, 1, 1) will be read as 0
 * - "wrap": sampling locations (x, y, z) outside (-1, -1, -1) and (1, 1, 1) will be read wrapped back according to ((x + 1) mod 2 - 1, (y + 1) mod 2 - 1, (z + 1) mod 2 - 1)
 */
at::Tensor GridSample3D_CPU(const at::Tensor &input, const at::Tensor &grid, const std::string &addressModeX,
                            const std::string &addressModeY, const std::string &addressModeZ,
                            c10::optional<at::Tensor> out);

/**
 * @ingroup pytorch_functions
 * @brief An implementation of reg23::GridSample3D_CPU that uses CUDA parallelisation.
 */
__host__ at::Tensor GridSample3D_CUDA(const at::Tensor &input, const at::Tensor &grid, const std::string &addressModeX,
                                      const std::string &addressModeY, const std::string &addressModeZ,
                                      c10::optional<at::Tensor> out);

/**
 * @tparam texture_t Type of the texture object that input data will be converted to for sampling.
 *
 * This struct is used as a namespace for code that is shared between different implementations of `GridSample3D_...`
 * functions
 */
template <typename texture_t> struct GridSample3D {

	static_assert(texture_t::DIMENSIONALITY == 3);

	using IntType = typename texture_t::IntType;
	using FloatType = typename texture_t::FloatType;
	using SizeType = typename texture_t::SizeType;
	using VectorType = typename texture_t::VectorType;
	using AddressModeType = typename texture_t::AddressModeType;

	struct CommonData {
		texture_t inputTexture{};
		at::Tensor flatOutput{};
	};

	__host__ static CommonData Common(const at::Tensor &input, const at::Tensor &grid, const std::string &addressModeX,
	                                  const std::string &addressModeY, const std::string &addressModeZ,
	                                  at::DeviceType device, c10::optional<at::Tensor> out) {
		// input should be a 3D tensor of floats on the chosen device
		TORCH_CHECK(input.sizes().size() == 3)
		TORCH_CHECK(input.dtype() == at::kFloat)
		TORCH_INTERNAL_ASSERT(input.device().type() == device)
		// grid should be a tensor of floats with a final dimension of 3 on the chosen device
		TORCH_CHECK(grid.sizes().back() == 3);
		TORCH_CHECK(grid.dtype() == at::kFloat);
		TORCH_INTERNAL_ASSERT(grid.device().type() == device)
		if (out) {
			// out should be a tensor of floats matching all but the last dimension of grid in size, on the chosen device
			TORCH_CHECK(out.value().sizes() == grid.sizes().slice(0, grid.sizes().size() - 1))
			TORCH_CHECK(out.value().dtype() == at::kFloat)
			TORCH_INTERNAL_ASSERT(out.value().device().type() == device)
		}

		// All addressMode<dim>s should be one of the valid values:
		AddressModeType addressModes = StringsToAddressModes<3>({{addressModeX, addressModeY, addressModeZ}});

		CommonData ret{};
		const SizeType inputSize = SizeType::FromIntArrayRef(input.sizes()).Flipped();
		const VectorType inputSpacing = 2.0 / inputSize.template StaticCast<FloatType>();
		ret.inputTexture = texture_t::FromTensor(input, inputSpacing, VectorType::Full(0.0), std::move(addressModes));
		ret.flatOutput = out
			                 ? out.value().view({-1})
			                 : torch::zeros(at::IntArrayRef({grid.numel() / 3}), input.contiguous().options());
		return ret;
	}
};

} // namespace reg23