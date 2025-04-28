#pragma once

#include "Common.h"
#include "Vec.h"

namespace ExtensionTest {

/**
 * @brief Compute an approximation of the Radon transform of the given 3D volume
 * @param volume The input volume to take the Radon transform of
 * @param volumeSpacing The spacing bewteen the volume layers in each cartsian direction
 * @param phiValues
 * @param thetaValues
 * @param rValues
 * @param samplesPerDirection
 * @return
 */
at::Tensor Radon3D_CPU(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                       const at::Tensor &thetaValues, const at::Tensor &rValues, int64_t samplesPerDirection);

at::Tensor DRadon3DDR_CPU(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                          const at::Tensor &thetaValues, const at::Tensor &rValues, int64_t samplesPerDirection);

__host__ at::Tensor Radon3D_CUDA(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                                 const at::Tensor &thetaValues, const at::Tensor &rValues, int64_t samplesPerDirection);

__host__ at::Tensor Radon3D_CUDA_V2(const at::Tensor &volume, const at::Tensor &volumeSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, int64_t samplesPerDirection);

__host__ at::Tensor DRadon3DDR_CUDA(const at::Tensor &volume, const at::Tensor &volumeSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, int64_t samplesPerDirection);

__host__ at::Tensor DRadon3DDR_CUDA_V2(const at::Tensor &volume, const at::Tensor &volumeSpacing,
                                       const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                       const at::Tensor &rValues, int64_t samplesPerDirection);

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

	struct CommonData {
		texture_t inputTexture{};
		Linear<Vec<double, 3> > mappingIndexToOffset{};
		double scaleFactor{};
		at::Tensor flatOutput{};
	};

	__host__ static CommonData Common(const at::Tensor &volume, const at::Tensor &volumeSpacing,
	                                  const at::Tensor &phiValues, const at::Tensor &thetaValues,
	                                  const at::Tensor &rValues, int64_t samplesPerDirection, at::DeviceType device) {
		// volume should be a 3D array of floats on the chosen device
		TORCH_CHECK(volume.sizes().size() == 3);
		TORCH_CHECK(volume.dtype() == at::kFloat);
		TORCH_INTERNAL_ASSERT(volume.device().type() == device);
		// volumeSpacing should be a 1D tensor of 3 floats or doubles
		TORCH_CHECK(volumeSpacing.sizes() == at::IntArrayRef{3});
		TORCH_CHECK(volumeSpacing.dtype() == at::kFloat || volumeSpacing.dtype() == at::kDouble);
		// phiValues, thetaValues and rValues should have matching sizes, contain floats, and be on the chosen device
		TORCH_CHECK(phiValues.sizes() == thetaValues.sizes());
		TORCH_CHECK(thetaValues.sizes() == rValues.sizes());
		TORCH_CHECK(phiValues.dtype() == at::kFloat);
		TORCH_CHECK(thetaValues.dtype() == at::kFloat);
		TORCH_CHECK(rValues.dtype() == at::kFloat);
		TORCH_INTERNAL_ASSERT(phiValues.device().type() == device);
		TORCH_INTERNAL_ASSERT(thetaValues.device().type() == device);
		TORCH_INTERNAL_ASSERT(rValues.device().type() == device);

		CommonData ret{};
		ret.inputTexture = texture_t::FromTensor(volume, volumeSpacing);
		const double planeSize = sqrtf(ret.inputTexture.SizeWorld().template Apply<double>(&Square<double>).Sum());
		ret.mappingIndexToOffset = GetMappingIToOffset(planeSize, samplesPerDirection);
		const double rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
		ret.scaleFactor = rootScaleFactor * rootScaleFactor;
		ret.flatOutput = torch::zeros(at::IntArrayRef({phiValues.numel()}), volume.contiguous().options());
		return ret;
	}

	__host__ __device__ [[nodiscard]] static Linear<Vec<double, 3> > GetMappingIToOffset(
		double planeSize, int64_t samplesPerDirection) {

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
	                                                               mappingIndexToTexCoord,
	                                                               int64_t samplesPerDirection) {
		float ret = 0.f;
		for (int64_t j = 0; j < samplesPerDirection; ++j) {
			for (int64_t i = 0; i < samplesPerDirection; ++i) {
				const double iF = static_cast<double>(i);
				const double jF = static_cast<double>(j);
				ret += texture.Sample(mappingIndexToTexCoord(Vec<double, 3>::Full(iF), Vec<double, 3>::Full(jF)));
			}
		}
		return ret;
	}

	__host__ __device__ [[nodiscard]] static float DIntegrateLoopedDMappingParameter(const texture_t &texture,
		const Linear2<Vec<double, 3> > &mappingIndexToTexCoord, const Vec<double, 3> &dTexCoordDR,
		int64_t samplesPerDirection) {
		float ret = 0.f;
		for (int64_t j = 0; j < samplesPerDirection; ++j) {
			for (int64_t i = 0; i < samplesPerDirection; ++i) {
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