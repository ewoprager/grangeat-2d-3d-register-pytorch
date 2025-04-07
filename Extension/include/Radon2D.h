#pragma once

#include "Common.h"
#include "Vec.h"

namespace ExtensionTest {

at::Tensor Radon2D_CPU(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
                       const at::Tensor &rValues, int64_t samplesPerLine);

at::Tensor DRadon2DDR_CPU(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
                          const at::Tensor &rValues, int64_t samplesPerLine);

__host__ at::Tensor Radon2D_CUDA(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
                                 const at::Tensor &rValues, int64_t samplesPerLine);

__host__ at::Tensor Radon2D_CUDA_V2(const at::Tensor &image, const at::Tensor &imageSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &rValues, int64_t samplesPerLine);

__host__ at::Tensor DRadon2DDR_CUDA(const at::Tensor &image, const at::Tensor &imageSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &rValues, int64_t samplesPerLine);

/**
 * Cartesian coordinates:
 *	- Origin at centre of image
 *	- x is to the right
 *	- y is up
 *
 * Radial coordinates:
 *	- Origin at centre of image
 *	- r is distance from origin
 *	- phi is radians anticlockwise from the positive x-direction
 *
 * @tparam texture_t
 */
template <typename texture_t> struct Radon2D {

	struct CommonData {
		texture_t inputTexture{};
		Linear<Vec<double, 2> > mappingIndexToOffset{};
		double scaleFactor{};
		at::Tensor flatOutput{};
	};

	__host__ static CommonData Common(const at::Tensor &image, const at::Tensor &imageSpacing,
	                                  const at::Tensor &phiValues, const at::Tensor &rValues, int64_t samplesPerLine,
	                                  at::DeviceType device) {
		// image should be a 2D tensor of floats on the chosen device
		TORCH_CHECK(image.sizes().size() == 2);
		TORCH_CHECK(image.dtype() == at::kFloat);
		TORCH_INTERNAL_ASSERT(image.device().type() == device);
		// imageSpacing should be a 1D tensor of 2 floats or doubles
		TORCH_CHECK(imageSpacing.sizes() == at::IntArrayRef{2});
		TORCH_CHECK(imageSpacing.dtype() == at::kFloat || imageSpacing.dtype() == at::kDouble);
		// phiValues and rValues should have the same sizes, should contain floats, and be on the chosen device
		TORCH_CHECK(phiValues.sizes() == rValues.sizes());
		TORCH_CHECK(phiValues.dtype() == at::kFloat);
		TORCH_CHECK(rValues.dtype() == at::kFloat);
		TORCH_INTERNAL_ASSERT(phiValues.device().type() == device);
		TORCH_INTERNAL_ASSERT(rValues.device().type() == device);

		CommonData ret{};
		ret.inputTexture = texture_t::FromTensor(image, imageSpacing);
		const double lineLength = sqrt(ret.inputTexture.SizeWorld().template Apply<double>(&Square<double>).Sum());
		ret.mappingIndexToOffset = GetMappingIToOffset(lineLength, samplesPerLine);
		ret.scaleFactor = lineLength / static_cast<float>(samplesPerLine);
		ret.flatOutput = torch::zeros(at::IntArrayRef({phiValues.numel()}), image.contiguous().options());
		return ret;
	}

	__host__ __device__ [[nodiscard]] static Linear<Vec<double, 2> > GetMappingIToOffset(
		double lineLength, int64_t samplesPerLine) {
		return {Vec<double, 2>::Full(-.5f * lineLength),
		        Vec<double, 2>::Full(lineLength / static_cast<double>(samplesPerLine - 1))};
	}

	__host__ __device__ [[nodiscard]] static Linear<Vec<double, 2> > GetMappingIndexToTexCoord(
		const texture_t &textureIn, double phi, double r, const Linear<Vec<double, 2> > &mappingIToOffset) {
		const double s = sin(phi);
		const double c = cos(phi);
		const Linear<Vec<double, 2> > mappingOffsetToWorld{{r * c, r * s}, {-s, c}};
		return textureIn.MappingWorldToTexCoord()(mappingOffsetToWorld(mappingIToOffset));
	}

	__host__ __device__ [[nodiscard]] static Vec<double, 2> GetDTexCoordDR(
		const texture_t &textureIn, double phi, double r) {
		const double sign = static_cast<double>(r > 0.) - static_cast<double>(r < 0.);
		const double s = sin(phi);
		const double c = cos(phi);
		return textureIn.MappingWorldToTexCoord().gradient * sign * Vec<double, 2>{c, s};
	}

	__host__ __device__ [[nodiscard]] static float IntegrateLooped(const texture_t &texture,
	                                                               const Linear<Vec<double, 2> > &
	                                                               mappingIndexToTexCoord, int64_t samplesPerLine) {
		float ret = 0.f;
		for (int64_t i = 0; i < samplesPerLine; ++i) {
			const double iF = static_cast<double>(i);
			ret += texture.Sample(mappingIndexToTexCoord(Vec<double, 2>::Full(iF)));
		}
		return ret;
	}

	__host__ __device__ [[nodiscard]] static float DIntegrateLoopedDMappingParameter(const texture_t &texture,
		const Linear<Vec<double, 2> > &mappingIndexToTexCoord, const Vec<double, 2> &dTexCoordDR,
		int64_t samplesPerLine) {
		float ret = 0.f;
		for (int64_t i = 0; i < samplesPerLine; ++i) {
			const double iF = static_cast<double>(i);
			const Vec<double, 2> texCoord = mappingIndexToTexCoord(Vec<double, 2>::Full(iF));
			ret += texture.DSampleDX(texCoord) * dTexCoordDR.X() + texture.DSampleDY(texCoord) * dTexCoordDR.Y();
		}
		return ret;
	}

};

} // namespace ExtensionTest