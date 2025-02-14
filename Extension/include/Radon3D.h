#pragma once

#include "Common.h"

namespace ExtensionTest {

at::Tensor radon3d_cpu(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                       const at::Tensor &phiValues, const at::Tensor &thetaValues, const at::Tensor &rValues,
                       long samplesPerDirection);

at::Tensor dRadon3dDR_cpu(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                          const at::Tensor &phiValues, const at::Tensor &thetaValues, const at::Tensor &rValues,
                          long samplesPerDirection);

__host__ at::Tensor radon3d_cuda(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                                 const at::Tensor &phiValues, const at::Tensor &thetaValues, const at::Tensor &rValues,
                                 long samplesPerDirection);

__host__ at::Tensor dRadon3dDR_cuda(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, long samplesPerDirection);

__host__ at::Tensor radon3d_v2_cuda(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, long samplesPerDirection);

__host__ at::Tensor dRadon3dDR_v2_cuda(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                                       const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                       const at::Tensor &rValues, long samplesPerDirection);

/**
 * Cartesian coordinates:
 *	- Origin at centre of volume
 *	- x, y and z are right-handed
 *
 * Radial coordinates:
 *	- Origin at centre of image
 *	- r is distance from origin
 *	- theta is radians left-hand-rule around y-axis from the positive x-direction
 *	- phi is radians right-hand-rule around z-axis from the positive x-direction
 *
 * @tparam texture_t
 */
template <typename texture_t> struct Radon3D {

	__host__ __device__ [[nodiscard]] static Linear<Vec<float, 3> > GetMappingIToOffset(
		float planeSize, long samplesPerDirection) {

		return {Vec<float, 3>::Full(-.5f * planeSize),
		        Vec<float, 3>::Full(planeSize / static_cast<float>(samplesPerDirection - 1))};
	}

	__host__ __device__ [[nodiscard]] static Linear2<Vec<float, 3> > GetMappingIndexToTexCoord(
		const texture_t &textureIn, float phi, float theta, float r, const Linear<Vec<float, 3> > &mappingIToOffset) {
		const float sp = sinf(phi);
		const float cp = cosf(phi);
		const float st = sinf(theta);
		const float ct = cosf(theta);
		const Linear2<Vec<float, 3> > mappingOffsetToWorld{{r * ct * cp, r * ct * sp, r * st}, {-sp, cp, 0.f},
		                                                   {-st * cp, -st * sp, ct}};
		return textureIn.MappingWorldToTexCoord()(mappingOffsetToWorld(mappingIToOffset));
	}

	__host__ __device__ [[nodiscard]] static Vec<float, 3> GetDTexCoordDR(
		const texture_t &textureIn, float phi, float theta, float r) {
		const float sign = static_cast<float>(r > 0.f) - static_cast<float>(r < 0.f);
		const float sp = sinf(phi);
		const float cp = cosf(phi);
		const float st = sinf(theta);
		const float ct = cosf(theta);
		return textureIn.MappingWorldToTexCoord().gradient * sign * Vec<float, 3>{ct * cp, ct * sp, st};
	}

	__host__ __device__ [[nodiscard]] static float IntegrateLooped(const texture_t &texture,
	                                                               const Linear2<Vec<float, 3> > &
	                                                               mappingIndexToTexCoord, long samplesPerDirection) {
		float ret = 0.f;
		for (long j = 0; j < samplesPerDirection; ++j) {
			for (long i = 0; i < samplesPerDirection; ++i) {
				const float iF = static_cast<float>(i);
				const float jF = static_cast<float>(j);
				ret += texture.Sample(mappingIndexToTexCoord(Vec<float, 3>::Full(iF), Vec<float, 3>::Full(jF)));
			}
		}
		return ret;
	}

	__host__ __device__ [[nodiscard]] static float DIntegrateLoopedDMappingParameter(const texture_t &texture,
		const Linear2<Vec<float, 3> > &mappingIndexToTexCoord, const Vec<float, 3> &dTexCoordDR,
		long samplesPerDirection) {
		float ret = 0.f;
		for (long j = 0; j < samplesPerDirection; ++j) {
			for (long i = 0; i < samplesPerDirection; ++i) {
				const float iF = static_cast<float>(i);
				const float jF = static_cast<float>(j);
				const Vec<float, 3> texCoord = mappingIndexToTexCoord(Vec<float, 3>::Full(iF), Vec<float, 3>::Full(jF));
				ret += texture.SampleXDerivative(texCoord) * dTexCoordDR.X() + texture.SampleYDerivative(texCoord) *
					dTexCoordDR.Y() + texture.SampleZDerivative(texCoord) * dTexCoordDR.Z();
			}
		}
		return ret;
	}

};

} // namespace ExtensionTest