#pragma once

#include "Common.h"

namespace reg23 {

/**
 * @ingroup pytorch_functions
 * @brief Compute an approximation of the Radon transform of the given 3D volume
 * @param volume a tensor of size (P,Q,R) containing `torch.float32`s: The input volume to take the Radon transform of
 * @param volumeSpacing a tensor of size (3,): The spacing between the volume layers in each cartesian direction
 * @param phiValues a tensor of any size containing `torch.float32`s: The values of the spherical coordinate $phi$ at
 * which to calculate approximations for plane integrals through the given volume
 * @param thetaValues a tensor of the same size as `phiValues` containing `torch.float32`s: The values of the spherical
 * coordinate $theta$ at which to calculate approximations for plane integrals through the given volume
 * @param rValues a tensor of the same size as `phiValues` containing `torch.float32`s: The values of the spherical
 * coordinate $r$ at which to calculate approximations for plane integrals through the given volume
 * @param samplesPerDirection The square-root of the number of samples to take from the volume for approximating each
 * plane integral
 * @return a tensor of the same size as `phiValues` containing `torch.float32`s: Approximations of the plane integrals
 * through the given volume at the given spherical coordinate locations
 *
 * The spherical coordinates should be defined according to the convention stated in Extension/Conventions.md
 */
at::Tensor Radon3D_CPU(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                       const at::Tensor &thetaValues, const at::Tensor &rValues, int64_t samplesPerDirection);

/**
 * @ingroup pytorch_functions
 * @brief Compute the derivative with respect to plane-origin distance of an approximation of the Radon transform of the
 * given 3D volume
 * @param volume a tensor of size (P,Q,R) containing `torch.float32`s: The input volume to take the Radon transform of
 * @param volumeSpacing a tensor of size (3,): The spacing between the volume layers in each cartesian direction
 * @param phiValues a tensor of any size containing `torch.float32`s: The values of the spherical coordinate phi at
 * which to calculate the derivatives of approximations for plane integrals through the given volume
 * @param thetaValues a tensor of the same size as `phiValues` containing `torch.float32`s: The values of the spherical
 * coordinate theta at which to calculate the derivatives of approximations for plane integrals through the given volume
 * @param rValues a tensor of the same size as `phiValues` containing `torch.float32`s: The values of the spherical
 * coordinate r at which to calculate the derivatives of approximations for plane integrals through the given volume
 * @param samplesPerDirection The square-root of the number of samples to take from the volume for approximating each
 * plane integral
 * @return a tensor of the same size as `phiValues` containing `torch.float32`s: The derivatives with respect to
 * plane-origin distance of the approximations (according to the `Radon3D_...` functions) of the plane integrals through
 * the given volume at the given spherical coordinate locations
 *
 * The spherical coordinates should be defined according to the convention stated in Extension/Conventions.md
 */
at::Tensor DRadon3DDR_CPU(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                          const at::Tensor &thetaValues, const at::Tensor &rValues, int64_t samplesPerDirection);

/**
 * @ingroup pytorch_functions
 * @brief An implementation of reg23::Radon3D_CPU that uses CUDA parallelisation
 *
 * A single kernel launch is made, with each kernel calculating one plane integral approximation.
 */
__host__ at::Tensor Radon3D_CUDA(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                                 const at::Tensor &thetaValues, const at::Tensor &rValues, int64_t samplesPerDirection);

/**
 * @brief An implementation of reg23::Radon3D_CPU that uses CUDA parallelisation
 *
 * One kernel launch is made per plane integral, and each plane integral approximation is done by summing samples from
 * multiple kernels in log-time.
 */
__host__ at::Tensor Radon3D_CUDA_V2(const at::Tensor &volume, const at::Tensor &volumeSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, int64_t samplesPerDirection);

/**
 * @ingroup pytorch_functions
 * @brief An implementation of reg23::DRadon3DDR_CPU that uses CUDA parallelisation
 *
 * A single kernel launch is made, with each kernel calculating one plane integral approximation.
 */
__host__ at::Tensor DRadon3DDR_CUDA(const at::Tensor &volume, const at::Tensor &volumeSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, int64_t samplesPerDirection);

/**
 * @brief An implementation of reg23::DRadon3DDR_CPU that uses CUDA parallelisation
 *
 * One kernel launch is made per plane integral, and each plane integral approximation is done by summing samples from
 * multiple kernels in log-time.
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

	static_assert(texture_t::DIMENSIONALITY == 3);

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

	__host__ static CommonData Common(const at::Tensor &volume, const at::Tensor &volumeSpacing,
	                                  const at::Tensor &phiValues, const at::Tensor &thetaValues,
	                                  const at::Tensor &rValues, int64_t samplesPerDirection, at::DeviceType device) {
		// volume should be a 3D array of floats on the chosen device
		TORCH_CHECK(volume.sizes().size() == 3);
		TORCH_CHECK(volume.dtype() == at::kFloat);
		TORCH_INTERNAL_ASSERT(volume.device().type() == device);
		// volumeSpacing should be a 1D tensor of 3 doubles
		TORCH_CHECK(volumeSpacing.sizes() == at::IntArrayRef{3});
		TORCH_CHECK(volumeSpacing.dtype() == at::kDouble);
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
		ret.inputTexture = texture_t::FromTensor(volume, VectorType::FromTensor(volumeSpacing));
		const FloatType planeSize = sqrt(
			ret.inputTexture.SizeWorld().template Apply<FloatType>(&Square<FloatType>).Sum());
		ret.mappingIndexToOffset = GetMappingIToOffset(planeSize, samplesPerDirection);
		const FloatType rootScaleFactor = planeSize / static_cast<FloatType>(samplesPerDirection);
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
	__host__ __device__ [[nodiscard]] static Linear<VectorType> GetMappingIToOffset(FloatType planeSize,
		int64_t samplesPerDirection) {

		return {VectorType::Full(-.5 * planeSize),
		        VectorType::Full(planeSize / static_cast<FloatType>(samplesPerDirection - 1))};
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
	__host__ __device__ [[nodiscard]] static Linear2<VectorType>
	GetMappingIndexToTexCoord(const texture_t &textureIn, FloatType phi, FloatType theta, FloatType r,
	                          const Linear<VectorType> &mappingIToOffset) {
		const FloatType sp = sin(phi);
		const FloatType cp = cos(phi);
		const FloatType st = sin(theta);
		const FloatType ct = cos(theta);
		const Linear2<VectorType> mappingOffsetToWorld{{r * ct * cp, r * ct * sp, r * st}, {-sp, cp, 0.f},
		                                               {-st * cp, -st * sp, ct}};
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
	__host__ __device__ [[nodiscard]] static VectorType GetDTexCoordDR(const texture_t &textureIn, FloatType phi,
	                                                                   FloatType theta, FloatType r) {
		const FloatType sign = static_cast<FloatType>(r > 0.) - static_cast<FloatType>(r < 0.);
		const FloatType sp = sin(phi);
		const FloatType cp = cos(phi);
		const FloatType st = sin(theta);
		const FloatType ct = cos(theta);
		return textureIn.MappingWorldToTexCoord().gradient * sign * VectorType{ct * cp, ct * sp, st};
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
	IntegrateLooped(const texture_t &texture, const Linear2<VectorType> &mappingIndexToTexCoord,
	                int64_t samplesPerDirection) {
		float ret = 0.f;
		for (int64_t j = 0; j < samplesPerDirection; ++j) {
			for (int64_t i = 0; i < samplesPerDirection; ++i) {
				const FloatType iF = static_cast<FloatType>(i);
				const FloatType jF = static_cast<FloatType>(j);
				ret += texture.Sample(mappingIndexToTexCoord(VectorType::Full(iF), VectorType::Full(jF)));
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
	DIntegrateLoopedDMappingParameter(const texture_t &texture, const Linear2<VectorType> &mappingIndexToTexCoord,
	                                  const VectorType &dTexCoordDMappingParameter, int64_t samplesPerDirection) {
		float ret = 0.f;
		for (int64_t j = 0; j < samplesPerDirection; ++j) {
			for (int64_t i = 0; i < samplesPerDirection; ++i) {
				const FloatType iF = static_cast<FloatType>(i);
				const FloatType jF = static_cast<FloatType>(j);
				const VectorType texCoord = mappingIndexToTexCoord(VectorType::Full(iF), VectorType::Full(jF));
				ret += VecDot(texture.DSampleDTexCoord(texCoord), dTexCoordDMappingParameter);
			}
		}
		return ret;
	}
};

} // namespace reg23