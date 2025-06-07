#pragma once

#include "Texture3DCPU.h"

namespace reg23 {

/**
 * @ingroup textures
 * @brief A 3D texture stored for access by the CPU, structured for storing an even distribution of values over the
 * surface of S^2, according to the HEALPix mapping. ToDo: citation here
 *
 * Texture sampling is done using spherical coordinates $(r, \theta, \phi)$, but only the first dimension $r$ actually
 * corresponds to the texture coordinate in the volume data. Across two dimensions the values are stored according to a
 * rectangular tesselation of the HEALPix spherical projection.
 *
 * **This object does not own the data it provides access to, it just holds a pointer to it**
 *
 * Both copy and move-constructable.
 */
class SinogramClassic3DCPU : Texture3DCPU {
  public:
	using Base = Texture3DCPU;

	static constexpr FloatType THETA_RANGE_LOW = -.5 * M_PI;
	static constexpr FloatType PHI_RANGE_LOW = -.5 * M_PI;
	static constexpr FloatType PHI_RANGE_HIGH = .5 * M_PI;

	static constexpr FloatType THETA_RANGE_HIGH(IntType thetaCount) {
		return M_PI * (.5 - 1. / static_cast<FloatType>(thetaCount));
	}

	SinogramClassic3DCPU() = default;

	/**
	 * @brief Construct the texture with data.
	 * @param _ptr A pointer to the beginning of the data
	 * @param rSpacing The spacing in world coordinates of the layers in the $r$ direction
	 */
	SinogramClassic3DCPU(const float *_ptr, SizeType _size, VectorType _spacing, VectorType _rangeCentres,
						 AddressModeType _addressModes)
		: Base(_ptr, std::move(_size), std::move(_spacing), std::move(_rangeCentres), std::move(_addressModes)) {
		mappingRThetaPhiToTexCoord = MappingWorldToTexCoord();
	}

	// yes copy
	SinogramClassic3DCPU(const SinogramClassic3DCPU &) = default;

	SinogramClassic3DCPU &operator=(const SinogramClassic3DCPU &) = default;

	// yes move
	SinogramClassic3DCPU(SinogramClassic3DCPU &&) = default;

	SinogramClassic3DCPU &operator=(SinogramClassic3DCPU &&) = default;

	/**
	 * @param tensor A 3D tensor of `torch.float32`s containing values for evenly spacing spherical coordinate locations
	 * in $\phi$, $\theta$ and $r$. The order of the dimensions along which the coordinates should vary is $\phi$,
	 * $\theta$, $r$.
	 * @param rSpacing The spacing between each HEALPix sphere
	 * @return An instance of this texture object that points to the data in the given image
	 */
	static SinogramClassic3DCPU FromTensor(const at::Tensor &tensor, FloatType rSpacing) {
		// tensor should be a 3D array of floats
		TORCH_CHECK(tensor.sizes().size() == 3)
		TORCH_CHECK(tensor.dtype() == at::kFloat)
		const SizeType tensorSizeRThetaPhi = SizeType::FromIntArrayRef(tensor.sizes()).Flipped();
		const FloatType thetaRangeHigh = THETA_RANGE_HIGH(tensorSizeRThetaPhi[1]);
		const FloatType thetaSpacing =
			(thetaRangeHigh - THETA_RANGE_LOW) / static_cast<FloatType>(tensorSizeRThetaPhi[1] - 1);
		const FloatType phiSpacing =
			(PHI_RANGE_HIGH - PHI_RANGE_LOW) / static_cast<FloatType>(tensorSizeRThetaPhi[2] - 1);
		return {tensor.contiguous().data_ptr<float>(),
				tensorSizeRThetaPhi,
				{rSpacing, thetaSpacing, phiSpacing},
				{0.0, 0.5 * (THETA_RANGE_LOW + thetaRangeHigh), 0.5 * (PHI_RANGE_LOW + PHI_RANGE_HIGH)},
				{TextureAddressMode::ZERO, TextureAddressMode::ZERO, TextureAddressMode::WRAP}};
	}

	/**
	 * @param rThetaPhi The location at which to sample in spherical coordinates
	 * @return The sample from this texture at the given coordinates using trilinear interpolation
	 */
	[[nodiscard]] __host__ __device__ float Sample(const VectorType &rThetaPhi) const {
		return Base::Sample(mappingRThetaPhiToTexCoord(rThetaPhi));
	}

  private:
	Linear<VectorType> mappingRThetaPhiToTexCoord{};
};

} // namespace reg23