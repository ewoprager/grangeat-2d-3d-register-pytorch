#pragma once

#include "Common.h"
#include "Texture.h"

namespace reg23 {

/**
 * @ingroup textures
 * @brief A 3D texture stored for access by the CPU, structured for storing an even distribution of values over the
 * surface of S^2, according to the HEALPix mapping. ToDo: citation here
 * @tparam texture_t
 *
 * Texture sampling is done using spherical coordinates $(r, \theta, \phi)$, but only the first dimension $r$ actually
 * corresponds to the texture coordinate in the volume data. Across two dimensions the values are stored according to a
 * rectangular tesselation of the HEALPix spherical projection.
 *
 * **This object does not own the data it provides access to, it just holds a pointer to it**
 *
 * Both copy and move-constructable.
 */
template <typename texture_t> class SinogramClassic3D : texture_t {
public:
	using Base = texture_t;
	static_assert(Base::DIMENSIONALITY == 3, "The texture type of a `SinogramClassic3D` must have dimensionality 3.");
	using IntType = typename Base::IntType;
	using FloatType = typename Base::FloatType;
	using SizeType = typename Base::SizeType;
	using VectorType = typename Base::VectorType;
	using AddressModeType = typename Base::AddressModeType;

	static constexpr FloatType THETA_RANGE_LOW = -.5 * M_PI;
	static constexpr FloatType THETA_RANGE_HIGH = .5 * M_PI;
	static constexpr FloatType PHI_RANGE_LOW = -.5 * M_PI;
	static constexpr FloatType PHI_RANGE_HIGH = .5 * M_PI;

	SinogramClassic3D() = default;

	/**
	 * @brief Construct the texture with data.
	 * @param texture
	 */
	explicit SinogramClassic3D(Base &texture) : Base(std::move(texture)) {
		mappingRThetaPhiToTexCoord = Base::MappingWorldToTexCoord();
	}

	// yes copy
	SinogramClassic3D(const SinogramClassic3D &) = default;

	SinogramClassic3D &operator=(const SinogramClassic3D &) = default;

	// yes move
	SinogramClassic3D(SinogramClassic3D &&) = default;

	SinogramClassic3D &operator=(SinogramClassic3D &&) = default;

	/**
	 * @param tensor A 3D tensor of `torch.float32`s containing values for evenly spacing spherical coordinate locations
	 * in $\phi$, $\theta$ and $r$. The order of the dimensions along which the coordinates should vary is $\phi$,
	 * $\theta$, $r$.
	 * @param rSpacing The spacing between each HEALPix sphere
	 * @return An instance of this texture object that points to the data in the given image
	 */
	static SinogramClassic3D FromTensor(const at::Tensor &tensor, FloatType rSpacing) {
		// tensor should be a 3D array of floats
		TORCH_CHECK(tensor.sizes().size() == 3)
		TORCH_CHECK(tensor.dtype() == at::kFloat)
		const SizeType tensorSizeRThetaPhi = SizeType::FromIntArrayRef(tensor.sizes()).Flipped();
		const FloatType thetaSpacing = (THETA_RANGE_HIGH - THETA_RANGE_LOW) / static_cast<FloatType>(
			                               tensorSizeRThetaPhi[1] - 2);
		const FloatType phiSpacing = (PHI_RANGE_HIGH - PHI_RANGE_LOW) / static_cast<FloatType>(
			                             tensorSizeRThetaPhi[2] - 2);
		return {tensor.contiguous().data_ptr<float>(), tensorSizeRThetaPhi, {rSpacing, thetaSpacing, phiSpacing},
		        VectorType::Full(0), {TextureAddressMode::ZERO, TextureAddressMode::ZERO, TextureAddressMode::ZERO}};
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