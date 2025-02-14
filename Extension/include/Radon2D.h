#pragma once

#include "Common.h"

namespace ExtensionTest {

at::Tensor radon2d_cpu(const at::Tensor &image, double xSpacing, double ySpacing, const at::Tensor &phiValues,
                       const at::Tensor &rValues, long samplesPerLine);

at::Tensor radon2d_v2_cpu(const at::Tensor &image, double xSpacing, double ySpacing, const at::Tensor &phiValues,
                          const at::Tensor &rValues, long samplesPerLine);

at::Tensor dRadon2dDR_cpu(const at::Tensor &image, double xSpacing, double ySpacing, const at::Tensor &phiValues,
                          const at::Tensor &rValues, long samplesPerLine);

__host__ at::Tensor radon2d_cuda(const at::Tensor &image, double xSpacing, double ySpacing, const at::Tensor &phiValues,
                                 const at::Tensor &rValues, long samplesPerLine);

__host__ at::Tensor radon2d_v2_cuda(const at::Tensor &image, double xSpacing, double ySpacing,
                                    const at::Tensor &phiValues, const at::Tensor &rValues, long samplesPerLine);

__host__ at::Tensor dRadon2dDR_cuda(const at::Tensor &image, double xSpacing, double ySpacing,
                                    const at::Tensor &phiValues, const at::Tensor &rValues, long samplesPerLine);

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

	__host__ __device__ [[nodiscard]] static Linear<Vec<double, 2> > GetMappingIToOffset(
		double lineLength, long samplesPerLine) {
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
	                                                               mappingIndexToTexCoord, long samplesPerLine) {
		float ret = 0.f;
		for (long i = 0; i < samplesPerLine; ++i) {
			const double iF = static_cast<double>(i);
			ret += texture.Sample(mappingIndexToTexCoord(Vec<double, 2>::Full(iF)));
		}
		return ret;
	}

	__host__ __device__ [[nodiscard]] static float DIntegrateLoopedDMappingParameter(const texture_t &texture,
		const Linear<Vec<double, 2> > &mappingIndexToTexCoord, const Vec<double, 2> &dTexCoordDR, long samplesPerLine) {
		float ret = 0.f;
		for (long i = 0; i < samplesPerLine; ++i) {
			const double iF = static_cast<double>(i);
			const Vec<double, 2> texCoord = mappingIndexToTexCoord(Vec<double, 2>::Full(iF));
			ret += texture.SampleXDerivative(texCoord) * dTexCoordDR.X() + texture.SampleYDerivative(texCoord) *
				dTexCoordDR.Y();
		}
		return ret;
	}

};

} // namespace ExtensionTest