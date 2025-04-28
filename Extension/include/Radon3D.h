#pragma once

#include "Common.h"

namespace ExtensionTest {

/**
 * @brief Compute an approximation of the Radon transform of the given 3D volume
 * @param volume A 3D tensor of `torch.float32`s; the input volume to take the Radon transform of
 * @param volumeSpacing The spacing between the volume layers in each cartesian direction
 * @param phiValues A tensor of `torch.float32`s of any size; the values of the spherical coordinate phi at which to
 * calculate approximations for plane integrals through the given volume
 * @param thetaValues A tensor of `torch.float32`s of the same size as `phiValues`; the values of the spherical
 * coordinate theta at which to calculate approximations for plane integrals through the given volume
 * @param rValues A tensor of `torch.float32`s of the same size as `phiValues`; the values of the spherical
 * coordinate r at which to calculate approximations for plane integrals through the given volume
 * @param samplesPerDirection The square-root of the number of samples to take from the volume for approximating each
 * plane integral
 * @return A tensor of `torch.float32`s matching `phiValues` in size; approximations of the plane integrals through the
 * given volume at the given spherical coordinate locations
 *
 * The spherical coordinates should be defined according to the convention stated in Extension/Conventions.md
 */
at::Tensor Radon3D_CPU(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
					   const at::Tensor &thetaValues, const at::Tensor &rValues, int64_t samplesPerDirection);

/**
 * @brief Compute the derivative with respect to plane-origin distance of an approximation of the Radon transform of the
 * given 3D volume
 * @param volume A 3D tensor of `torch.float32`s; the input volume to take the Radon transform of
 * @param volumeSpacing The spacing between the volume layers in each cartesian direction
 * @param phiValues A tensor of `torch.float32`s of any size; the values of the spherical coordinate phi at which to
 * calculate the derivatives of approximations for plane integrals through the given volume
 * @param thetaValues A tensor of `torch.float32`s of the same size as `phiValues`; the values of the spherical
 * coordinate theta at which to calculate the derivatives of approximations for plane integrals through the given volume
 * @param rValues A tensor of `torch.float32`s of the same size as `phiValues`; the values of the spherical
 * coordinate r at which to calculate the derivatives of approximations for plane integrals through the given volume
 * @param samplesPerDirection The square-root of the number of samples to take from the volume for approximating each
 * plane integral
 * @return A tensor of `torch.float32`s matching `phiValues` in size; the derivatives with respect to plane-origin
 * distance of the approximations (according to the `Radon3D_...` functions) of the plane integrals through the given
 * volume at the given spherical coordinate locations
 *
 * The spherical coordinates should be defined according to the convention stated in Extension/Conventions.md
 */
at::Tensor DRadon3DDR_CPU(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
						  const at::Tensor &thetaValues, const at::Tensor &rValues, int64_t samplesPerDirection);

/**
 * An implementation of `Radon3D_CPU` that uses CUDA parallelisation. A single kernel launch is made, with each kernel
 * calculating one plane integral approximation.
 */
__host__ at::Tensor Radon3D_CUDA(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
								 const at::Tensor &thetaValues, const at::Tensor &rValues, int64_t samplesPerDirection);

/**
 * An implementation of `Radon3D_CPU` that uses CUDA parallelisation. One kernel launch is made per plane integral, and
 * each plane integral approximation is done by summing samples from multiple kernels in log-time.
 */
__host__ at::Tensor Radon3D_CUDA_V2(const at::Tensor &volume, const at::Tensor &volumeSpacing,
									const at::Tensor &phiValues, const at::Tensor &thetaValues,
									const at::Tensor &rValues, int64_t samplesPerDirection);

/**
 * An implementation of `DRadon3DDR_CPU` that uses CUDA parallelisation. A single kernel launch is made, with each
 * kernel calculating one plane integral approximation.
 */
__host__ at::Tensor DRadon3DDR_CUDA(const at::Tensor &volume, const at::Tensor &volumeSpacing,
									const at::Tensor &phiValues, const at::Tensor &thetaValues,
									const at::Tensor &rValues, int64_t samplesPerDirection);

/**
 * An implementation of `DRadon3DDR_CPU` that uses CUDA parallelisation. One kernel launch is made per plane integral,
 * and each plane integral approximation is done by summing samples from multiple kernels in log-time.
 */
__host__ at::Tensor DRadon3DDR_CUDA_V2(const at::Tensor &volume, const at::Tensor &volumeSpacing,
									   const at::Tensor &phiValues, const at::Tensor &thetaValues,
									   const at::Tensor &rValues, int64_t samplesPerDirection);

/**
 * @tparam texture_t Type of the texture object that input data will be converted to for sampling.
 *
 * This struct is used as a namespace for code that is shared between different implementations of `Radon3D_...` and
 * `DRadon3DDR_...` functions
 */
template <typename texture_t> struct Radon3D {

	struct CommonData {
		texture_t inputTexture{};
		Linear<Vec<double, 3>> mappingIndexToOffset{};
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

	/**
	 * @param planeSize
	 * @param samplesPerDirection
	 * @return The `Linear` mapping from integration iteration index to world sample distance, according to the given
	 * plane size and sample count
	 *
	 * This is used as part of the approximation of a plane integral: for each sample taken to contribute to the plane
	 * integral, the returned functor maps the sample index to the (untransformed) world offset.
	 *
	 * The mapping is generated for 3-vectors, as it will ultimately be applied to 3-vectors.
	 */
	__host__ __device__ [[nodiscard]] static Linear<Vec<double, 3>> GetMappingIToOffset(double planeSize,
																						int64_t samplesPerDirection) {

		return {Vec<double, 3>::Full(-.5f * planeSize),
				Vec<double, 3>::Full(planeSize / static_cast<double>(samplesPerDirection - 1))};
	}

	/**
	 * @param textureIn The texture object which the returned mapping will be used to access.
	 * @param phi The phi spherical coordinate of the plane this mapping should correspond to
	 * @param theta The theta spherical coordinate of the plane this mapping should correspond to
	 * @param r The r spherical coordinate of the plane this mapping should correspond to
	 * @param mappingIToOffset The mapping from integration iteration index to untransformed sampling offset
	 * @return The `Linear` mapping from integration iteration index to 3D cartesian texture coordinate sampling
	 * location, according to the given texture object and integration plane (parameterised in spherical coordinates)
	 *
	 * The spherical coordinates should be defined according to the convention stated in Extension/Conventions.md
	 */
	__host__ __device__ [[nodiscard]] static Linear2<Vec<double, 3>>
	GetMappingIndexToTexCoord(const texture_t &textureIn, double phi, double theta, double r,
							  const Linear<Vec<double, 3>> &mappingIToOffset) {
		const double sp = sin(phi);
		const double cp = cos(phi);
		const double st = sin(theta);
		const double ct = cos(theta);
		const Linear2<Vec<double, 3>> mappingOffsetToWorld{
			{r * ct * cp, r * ct * sp, r * st}, {-sp, cp, 0.f}, {-st * cp, -st * sp, ct}};
		return textureIn.MappingWorldToTexCoord()(mappingOffsetToWorld(mappingIToOffset));
	}

	/**
	 * @param textureIn The texture object which is being accessed
	 * @param phi The phi spherical coordinate at which the result should be evaluated
	 * @param theta The theta spherical coordinate at which the result should be evaluated
	 * @param r The r spherical coordinate at which the result should be evaluated
	 * @return The derivative of the 3D cartesian texture coordinate in the given texture at the given world position
	 * defined in spherical coordinates, with respect to the **unsigned** r spherical coordinate
	 *
	 * The spherical coordinates should be defined according to the convention stated in Extension/Conventions.md,
	 * however the parameter with respect to which this is evaluating the derivative is the **unsigned** version of `r`,
	 * i.e. the **unsigned** distance between the orign and the plane.
	 */
	__host__ __device__ [[nodiscard]] static Vec<double, 3> GetDTexCoordDR(const texture_t &textureIn, double phi,
																		   double theta, double r) {
		const double sign = static_cast<double>(r > 0.) - static_cast<double>(r < 0.);
		const double sp = sin(phi);
		const double cp = cos(phi);
		const double st = sin(theta);
		const double ct = cos(theta);
		return textureIn.MappingWorldToTexCoord().gradient * sign * Vec<double, 3>{ct * cp, ct * sp, st};
	}

	/**
	 * @param texture The texture object to sample
	 * @param mappingIndexToTexCoord The mapping from integration iteration index to 3D cartesian texture coordinate
	 * (this determines the integration plane)
	 * @param samplesPerDirection The square-root of the number of samples to make
	 * @return An approximation of the plane integral through the given texture, determined by the given index ->
	 * texture coordinate mapping. Samples are taken in a [samplesPerDirection x samplesPerDirection] square grid over
	 * the plane.
	 *
	 * `mappingIndexToTexCoord` can be determined using `GetMappingIndexToTexCoord` (here the plane to be integrated is
	 * defined in spherical coordinates). This function is separate as it is often more efficient to precalculate the
	 * value of `mappingIndexToTexCoord`.
	 */
	__host__ __device__ [[nodiscard]] static float
	IntegrateLooped(const texture_t &texture, const Linear2<Vec<double, 3>> &mappingIndexToTexCoord,
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

	/**
	 * @param texture The texture object to sample
	 * @param mappingIndexToTexCoord The mapping from integration iteration index to 3D cartesian texture coordinate
	 * (this determines the integration plane)
	 * @param dTexCoordDMappingParameter The derivative of the sampling locations with respect to some parameter. This
	 * is assumed to be constant over the plane.
	 * @param samplesPerDirection The square-root of the number of samples to make
	 * @return The derivative of the approximation of a plane integral as in `IntegrateLooped`, with respect to some
	 * parameter (according to the value of `dTexCoordDMappingParameter`).
	 */
	__host__ __device__ [[nodiscard]] static float
	DIntegrateLoopedDMappingParameter(const texture_t &texture, const Linear2<Vec<double, 3>> &mappingIndexToTexCoord,
									  const Vec<double, 3> &dTexCoordDMappingParameter, int64_t samplesPerDirection) {
		float ret = 0.f;
		for (int64_t j = 0; j < samplesPerDirection; ++j) {
			for (int64_t i = 0; i < samplesPerDirection; ++i) {
				const double iF = static_cast<double>(i);
				const double jF = static_cast<double>(j);
				const Vec<double, 3> texCoord =
					mappingIndexToTexCoord(Vec<double, 3>::Full(iF), Vec<double, 3>::Full(jF));
				ret += texture.DSampleDX(texCoord) * dTexCoordDMappingParameter.X() +
					   texture.DSampleDY(texCoord) * dTexCoordDMappingParameter.Y() +
					   texture.DSampleDZ(texCoord) * dTexCoordDMappingParameter.Z();
			}
		}
		return ret;
	}
};

} // namespace ExtensionTest