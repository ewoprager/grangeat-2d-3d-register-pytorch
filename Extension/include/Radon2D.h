/**
 * @file
 * @brief Global functions
 */

#pragma once

#include "Common.h"

namespace reg23 {

/**
 * @ingroup pytorch_functions
 * @brief Compute an approximation of the Radon transform of the given 2D image
 * @param image a tensor of size (N, M) containing `torch.float32`s: The input image to take the Radon transform of
 * @param imageSpacing a tensor of size (2,): The spacing between the image columns and rows
 * @param phiValues a tensor of any size containing `torch.float32`s: The values of the polar coordinate phi at which to
 * calculate approximations for line integrals through the given image
 * @param rValues a tensor of the same size as `phiValues` containing `torch.float32`s: The values of the polar
 * coordinate r at which to calculate approximations for line integrals through the given image
 * @param samplesPerLine The number of samples to take from the image for approximating each line integral
 * @return a tensor matching `phiValues` in size containing `torch.float32`s: Approximations of the line integrals
 * through the given image at the given polar coordinate locations
 *
 * The polar coordinates should be defined according to the convention stated in Extension/Conventions.md
 */
at::Tensor Radon2D_CPU(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
                       const at::Tensor &rValues, int64_t samplesPerLine);

/**
 * @ingroup pytorch_functions
 * @brief Compute the derivative with respect to plane-origin distance of an approximation of the Radon transform of the
 * given 2D image
 * @param image a tensor of size (N, M) containing `torch.float32`s: The input image to take the Radon transform of
 * @param imageSpacing a tensor of size (2,): The spacing between the image columns and rows
 * @param phiValues a tensor of any size containing `torch.float32`s: The values of the polar coordinate phi at which to
 * calculate approximations for line integrals through the given image
 * @param rValues a tensor of the same size as `phiValues` containing `torch.float32`s: The values of the polar
 * coordinate r at which to calculate approximations for line integrals through the given image
 * @param samplesPerLine The number of samples to take from the image for approximating each line integral
 * @return a tensor matching `phiValues` in size containing `torch.float32`s: The derivatives with respect to
 * plane-origin distance of the approximations (according to the `Radon2D_...` functions) of the line integrals through
 * the given image at the given polar coordinate locations
 *
 * The polar coordinates should be defined according to the convention stated in Extension/Conventions.md
 */
at::Tensor DRadon2DDR_CPU(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
                          const at::Tensor &rValues, int64_t samplesPerLine);

/**
 * @ingroup pytorch_functions
 * @brief An implementation of reg23::Radon2D_CPU that uses CUDA parallelisation
 *
 * A single kernel launch is made, with each kernel calculating one line integral approximation.
 */
__host__ at::Tensor Radon2D_CUDA(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
                                 const at::Tensor &rValues, int64_t samplesPerLine);

/**
 * @brief An implementation of reg23::Radon2D_CPU that uses CUDA parallelisation
 *
 * One kernel launch is made per line integral, and each line integral approximation is done by summing samples from
 * multiple kernels in log-time.
 */
__host__ at::Tensor Radon2D_CUDA_V2(const at::Tensor &image, const at::Tensor &imageSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &rValues, int64_t samplesPerLine);

/**
 * @ingroup pytorch_functions
 * @brief An implementation of reg23::DRadon2DDR_CPU that uses CUDA parallelisation
 *
 * A single kernel launch is made, with each kernel calculating one line integral approximation.
 */
__host__ at::Tensor DRadon2DDR_CUDA(const at::Tensor &image, const at::Tensor &imageSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &rValues, int64_t samplesPerLine);

/**
 * @tparam texture_t Type of the texture object that input data will be converted to for sampling.
 *
 * This struct is used as a namespace for code that is shared between different implementations of `Radon2D_...` and
 * `DRadon2DDR_...` functions
 */
template <typename texture_t> struct Radon2D {

	static_assert(texture_t::DIMENSIONALITY == 2);

	using IntType = typename texture_t::IntType;
	using FloatType = typename texture_t::FloatType;
	using SizeType = typename texture_t::SizeType;
	using VectorType = typename texture_t::VectorType;
	using AddressModeType = typename texture_t::AddressModeType;

	struct CommonData {
		texture_t inputTexture{};
		Linear<VectorType> mappingIndexToOffset{};
		FloatType scaleFactor{};
		at::Tensor flatOutput{};
	};

	__host__ static CommonData Common(const at::Tensor &image, const at::Tensor &imageSpacing,
	                                  const at::Tensor &phiValues, const at::Tensor &rValues, int64_t samplesPerLine,
	                                  at::DeviceType device) {
		// image should be a 2D tensor of floats on the chosen device
		TORCH_CHECK(image.sizes().size() == 2);
		TORCH_CHECK(image.dtype() == at::kFloat);
		TORCH_INTERNAL_ASSERT(image.device().type() == device);
		// imageSpacing should be a 1D tensor of 2 doubles
		TORCH_CHECK(imageSpacing.sizes() == at::IntArrayRef{2});
		TORCH_CHECK(imageSpacing.dtype() == at::kDouble);
		// phiValues and rValues should have the same sizes, should contain floats, and be on the chosen device
		TORCH_CHECK(phiValues.sizes() == rValues.sizes());
		TORCH_CHECK(phiValues.dtype() == at::kFloat);
		TORCH_CHECK(rValues.dtype() == at::kFloat);
		TORCH_INTERNAL_ASSERT(phiValues.device().type() == device);
		TORCH_INTERNAL_ASSERT(rValues.device().type() == device);

		CommonData ret{};
		ret.inputTexture = texture_t::FromTensor(image, VectorType::FromTensor(imageSpacing));
		const FloatType lineLength = sqrt(
			ret.inputTexture.SizeWorld().template Apply<FloatType>(&Square<FloatType>).Sum());
		ret.mappingIndexToOffset = GetMappingIToOffset(lineLength, samplesPerLine);
		ret.scaleFactor = lineLength / static_cast<float>(samplesPerLine);
		ret.flatOutput = torch::zeros(at::IntArrayRef({phiValues.numel()}), image.contiguous().options());
		return ret;
	}

	/**
	 * @param lineLength
	 * @param samplesPerLine
	 * @return The `Linear` mapping from integration iteration index to world sample distance, according to the given
	 * line length and sample count
	 *
	 * This is used as part of the approximation of a line integral: for each sample taken to contribute to the line
	 * integral, the returned functor maps the sample index to the (untransformed) world offset.
	 *
	 * The mapping is generated for 2-vectors, as it will ultimately be applied to 2-vectors.
	 */
	__host__ __device__ [[nodiscard]] static Linear<VectorType> GetMappingIToOffset(FloatType lineLength,
		int64_t samplesPerLine) {
		return {VectorType::Full(-.5f * lineLength),
		        VectorType::Full(lineLength / static_cast<FloatType>(samplesPerLine - 1))};
	}

	/**
	 * @param textureIn The texture object which the returned mapping will be used to access.
	 * @param phi The phi polar coordinate of the plane this mapping should correspond to
	 * @param r The r polar coordinate of the plane this mapping should correspond to
	 * @param mappingIToOffset The mapping from integration iteration index to untransformed sampling offset
	 * @return The `Linear` mapping from integration iteration index to 2D cartesian texture coordinate sampling
	 * location, according to the given texture object and integration line (parameterised in polar coordinates)
	 *
	 * The polar coordinates should be defined according to the convention stated in Extension/Conventions.md
	 */
	__host__ __device__ [[nodiscard]] static Linear<VectorType>
	GetMappingIndexToTexCoord(const texture_t &textureIn, FloatType phi, FloatType r,
	                          const Linear<VectorType> &mappingIToOffset) {
		const FloatType s = sin(phi);
		const FloatType c = cos(phi);
		const Linear<VectorType> mappingOffsetToWorld{{r * c, r * s}, {-s, c}};
		return textureIn.MappingWorldToTexCoord()(mappingOffsetToWorld(mappingIToOffset));
	}

	/**
	 * @param textureIn The texture object which is being accessed
	 * @param phi The phi polar coordinate at which the result should be evaluated
	 * @param r The r polar coordinate at which the result should be evaluated
	 * @return The derivative of the 2D cartesian texture coordinate in the given texture at the given world position
	 * defined in polar coordinates, with respect to the **unsigned** r polar coordinate
	 *
	 * The polar coordinates should be defined according to the convention stated in Extension/Conventions.md, however
	 * the parameter with respect to which this is evaluating the derivative is the **unsigned** version of `r`, i.e.
	 * the **unsigned** distance between the orign and the line.
	 */
	__host__ __device__ [[nodiscard]] static VectorType GetDTexCoordDR(const texture_t &textureIn, FloatType phi,
	                                                                   FloatType r) {
		const FloatType sign = static_cast<FloatType>(r > 0.) - static_cast<FloatType>(r < 0.);
		const FloatType s = sin(phi);
		const FloatType c = cos(phi);
		return textureIn.MappingWorldToTexCoord().gradient * sign * VectorType{c, s};
	}

	/**
	 * @param texture The texture object to sample
	 * @param mappingIndexToTexCoord The mapping from integration iteration index to 2D cartesian texture coordinate
	 * (this determines the integration line)
	 * @param samplesPerLine The number of samples to make
	 * @return An approximation of the line integral through the given texture, determined by the given index ->
	 * texture coordinate mapping. `samplesPerLine` samples are taken evenly spaced along the line.
	 *
	 * `mappingIndexToTexCoord` can be determined using `GetMappingIndexToTexCoord` (here the line to be integrated is
	 * defined in polar coordinates). This function is separate as it is often more efficient to precalculate the value
	 * of `mappingIndexToTexCoord`.
	 */
	__host__ __device__ [[nodiscard]] static float IntegrateLooped(const texture_t &texture,
	                                                               const Linear<VectorType> &mappingIndexToTexCoord,
	                                                               int64_t samplesPerLine) {
		float ret = 0.f;
		for (int64_t i = 0; i < samplesPerLine; ++i) {
			const FloatType iF = static_cast<FloatType>(i);
			ret += texture.Sample(mappingIndexToTexCoord(VectorType::Full(iF)));
		}
		return ret;
	}

	/**
	 * @param texture The texture object to sample
	 * @param mappingIndexToTexCoord The mapping from integration iteration index to 2D cartesian texture coordinate
	 * (this determines the integration line)
	 * @param dTexCoordDMappingParameter The derivative of the sampling locations with respect to some parameter. This
	 * is assumed to be constant along the line.
	 * @param samplesPerLine The number of samples to make
	 * @return The derivative of the approximation of a line integral as in `IntegrateLooped`, with respect to some
	 * parameter (according to the value of `dTexCoordDMappingParameter`).
	 */
	__host__ __device__ [[nodiscard]] static float
	DIntegrateLoopedDMappingParameter(const texture_t &texture, const Linear<VectorType> &mappingIndexToTexCoord,
	                                  const VectorType &dTexCoordDMappingParameter, int64_t samplesPerLine) {
		float ret = 0.f;
		for (int64_t i = 0; i < samplesPerLine; ++i) {
			const FloatType iF = static_cast<FloatType>(i);
			const VectorType texCoord = mappingIndexToTexCoord(VectorType::Full(iF));
			ret += texture.DSampleDX(texCoord) * dTexCoordDMappingParameter.X() + texture.DSampleDY(texCoord) *
				dTexCoordDMappingParameter.Y();
		}
		return ret;
	}
};

} // namespace reg23