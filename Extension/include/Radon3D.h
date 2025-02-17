#pragma once

#include "Common.h"
#include "Vec.h"

namespace ExtensionTest {

at::Tensor radon3d_cpu(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                       const at::Tensor &thetaValues, const at::Tensor &rValues, long samplesPerDirection);

at::Tensor dRadon3dDR_cpu(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                          const at::Tensor &thetaValues, const at::Tensor &rValues, long samplesPerDirection);

__host__ at::Tensor radon3d_cuda(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                                 const at::Tensor &thetaValues, const at::Tensor &rValues, long samplesPerDirection);

__host__ at::Tensor dRadon3dDR_cuda(const at::Tensor &volume, const at::Tensor &volumeSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, long samplesPerDirection);

__host__ at::Tensor radon3d_v2_cuda(const at::Tensor &volume, const at::Tensor &volumeSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, long samplesPerDirection);

__host__ at::Tensor dRadon3dDR_v2_cuda(const at::Tensor &volume, const at::Tensor &volumeSpacing,
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

	__host__ __device__ [[nodiscard]] static Linear<Vec<double, 3> > GetMappingIToOffset(
		double planeSize, long samplesPerDirection) {

		return {Vec<double, 3>::Full(-.5f * planeSize),
		        Vec<double, 3>::Full(planeSize / static_cast<double>(samplesPerDirection - 1))};
	}

	__host__ __device__ [[nodiscard]] static Linear2<Vec<double, 3> > GetMappingIndexToTexCoord(
		const texture_t &textureIn, double phi, double theta, double r,
		const Linear<Vec<double, 3> > &mappingIToOffset) {
		const double sp = sin(phi);
		const double cp = cos(phi);
		const double st = sin(theta);
		const double ct = cos(theta);
		const Linear2<Vec<double, 3> > mappingOffsetToWorld{{r * ct * cp, r * ct * sp, r * st}, {-sp, cp, 0.f},
		                                                    {-st * cp, -st * sp, ct}};
		return textureIn.MappingWorldToTexCoord()(mappingOffsetToWorld(mappingIToOffset));
	}

	__host__ __device__ [[nodiscard]] static Vec<double, 3> GetDTexCoordDR(
		const texture_t &textureIn, double phi, double theta, double r) {
		const double sign = static_cast<double>(r > 0.) - static_cast<double>(r < 0.);
		const double sp = sin(phi);
		const double cp = cos(phi);
		const double st = sin(theta);
		const double ct = cos(theta);
		return textureIn.MappingWorldToTexCoord().gradient * sign * Vec<double, 3>{ct * cp, ct * sp, st};
	}

	__host__ __device__ [[nodiscard]] static float IntegrateLooped(const texture_t &texture,
	                                                               const Linear2<Vec<double, 3> > &
	                                                               mappingIndexToTexCoord, long samplesPerDirection) {
		float ret = 0.f;
		for (long j = 0; j < samplesPerDirection; ++j) {
			for (long i = 0; i < samplesPerDirection; ++i) {
				const double iF = static_cast<double>(i);
				const double jF = static_cast<double>(j);
				ret += texture.Sample(mappingIndexToTexCoord(Vec<double, 3>::Full(iF), Vec<double, 3>::Full(jF)));
			}
		}
		return ret;
	}

	__host__ __device__ [[nodiscard]] static float DIntegrateLoopedDMappingParameter(const texture_t &texture,
		const Linear2<Vec<double, 3> > &mappingIndexToTexCoord, const Vec<double, 3> &dTexCoordDR,
		long samplesPerDirection) {
		float ret = 0.f;
		for (long j = 0; j < samplesPerDirection; ++j) {
			for (long i = 0; i < samplesPerDirection; ++i) {
				const double iF = static_cast<double>(i);
				const double jF = static_cast<double>(j);
				const Vec<double, 3> texCoord = mappingIndexToTexCoord(Vec<double, 3>::Full(iF),
				                                                       Vec<double, 3>::Full(jF));
				ret += texture.DSampleDX(texCoord) * dTexCoordDR.X() + texture.DSampleDY(texCoord) * dTexCoordDR.Y() +
					texture.DSampleDZ(texCoord) * dTexCoordDR.Z();
			}
		}
		return ret;
	}

};

} // namespace ExtensionTest